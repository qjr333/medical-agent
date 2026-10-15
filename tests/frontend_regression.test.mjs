// Run with: node --test tests/frontend_regression.test.mjs
// Executes the actual page scripts/compiled React component with local DOM,
// network and hooks doubles. No browser, API server or live model is used.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'
import test from 'node:test'
import { transformWithOxc } from '../frontend/node_modules/vite/dist/node/index.js'

const root = resolve(fileURLToPath(new URL('..', import.meta.url)))
const flush = () => new Promise((done) => setImmediate(done))
const response = (data) => ({ ok: true, text: async () => JSON.stringify(data), json: async () => data })

function page(file) {
  const elements = new Map()
  function element() {
    return { children: [], style: {}, dataset: {}, textContent: '', value: '',
      append(...children) { this.children.push(...children) },
      appendChild(child) { this.children.push(child); return child },
      replaceChildren(...children) { this.children = children },
      addEventListener() {}, remove() {}, focus() {},
      querySelector() { return this.children[0] || element() },
    }
  }
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, element())
    return elements.get(id)
  }
  const calls = [], timeouts = [], intervals = new Map()
  let intervalId = 0
  const context = vm.createContext({
    console, AbortController, performance,
    setTimeout(_callback, ms) { timeouts.push(ms); return timeouts.length },
    clearTimeout() {},
    setInterval(callback, ms) { intervals.set(++intervalId, { callback, ms }); return intervalId },
    clearInterval(id) { intervals.delete(id) },
    document: { getElementById: get, querySelectorAll: () => [], createElement: element },
    sessionStorage: { getItem: () => '[]', setItem() {} },
    fetch: async (url, options) => {
      calls.push({ url, options })
      return response(url === '/api/demo' ? { patient: { patient_name: '虚构患者' }, bp_history: [] } : { items: [] })
    },
  })
  // Script extraction is limited to the inline executable scripts, not copied
  // implementations of the functions under test.
  const source = readFileSync(resolve(root, file), 'utf8')
  for (const match of source.matchAll(/<script>([\s\S]*?)<\/script>/g)) vm.runInContext(match[1], context)
  return { context, get, calls, timeouts, intervals, run: (code) => vm.runInContext(code, context) }
}

test('HTML terminal states disable chat; not-yet-taken is not displayed as unknown', async () => {
  const ui = page('static/index.html')
  // Let initial demo loading settle, then exercise real response/control code.
  await flush()
  ui.run(`sessionId='fixture-session'; followupDone=normalizeResponse({state:{followup_status:'partially_completed'}}).done;
    updatePatientState({followup_status:'partially_completed',medication_status:'not_taken_yet'}); updateControls();`)
  assert.equal(ui.get('medicationStatus').textContent, '尚未服用')
  assert.equal(ui.get('statusBadge').textContent, '随访部分完成')
  assert.equal(ui.get('messageInput').disabled, true)
  assert.equal(ui.get('sendButton').disabled, true)
  const before = ui.calls.length
  await ui.run('sendMessage()')
  assert.equal(ui.calls.length, before)
  for (const status of ['completed', 'partially_completed', 'escalated']) {
    assert.equal(ui.run(`normalizeResponse({state:{followup_status:'${status}'}}).done`), true)
  }
  assert.equal(ui.run(`normalizeResponse({state:{followup_status:'waiting_for_user'}}).done`), false)
})

test('HTML waits for bounded model stages without auto-retrying or extending normal reads', async () => {
  const ui = page('static/index.html')
  await flush()
  ui.context.fetch = async (url, options) => {
    ui.calls.push({ url, options })
    return response({ session_id: 'fixture-session', message: '已收到。', state: { followup_status: 'waiting_for_user' } })
  }
  await ui.run(`triggerAgent('scheduled')`)
  assert.equal(ui.timeouts.at(-1), 180000)
  ui.run(`messageInput.value='已经吃了'`)
  await ui.run('sendMessage()')
  assert.equal(ui.timeouts.at(-1), 360000)
  assert.equal(ui.calls.filter((c) => c.url === '/api/trigger').length, 1)
  assert.equal(ui.calls.filter((c) => c.url === '/api/chat').length, 1)
  await ui.run(`fetchJSON('/api/health')`)
  assert.equal(ui.timeouts.at(-1), 30000)
})

test('V5 page shows only actual validation/retry/failure rows and safe details', async () => {
  const ui = page('static/v5-demo.html')
  await flush()
  ui.context.payload = { options: { model_mode: 'live' }, status: 'interrupted', real_model_requests: 2,
    transcript: [], trace: [
      { stage: 'information_validation', detail: { purpose: 'VERIFY_EXTRACT', status: 'error', failure_kind: 'timeout', duration_ms: 50000, reason: 'private-thought' } },
      { stage: 'model_retry', detail: { purpose: 'VERIFY_EXTRACT', attempt: 2, max_attempts: 2 } },
      { stage: 'execution_error', detail: { status: 'paused', recoverable: true, error_type: 'TimeoutError' } },
    ] }
  ui.run('draw(payload)')
  const rows = ui.get('flow').children
  assert.equal(rows.length, 3)
  assert.deepEqual(rows.map((row) => row.children[0].textContent), ['医学事实验证', '模型有限重试', '执行异常'])
  const displayed = JSON.stringify(rows)
  for (const text of ['超时', '50000 ms', '第 2/2 次', '可恢复', 'TimeoutError']) assert.ok(displayed.includes(text))
  assert.ok(!displayed.includes('private-thought'))
  ui.context.payload.trace = []
  ui.run('draw(payload)')
  assert.equal(ui.get('flow').children.length, 0)
})

test('Recoverable simulation pauses polling and resumes once only after a user click', async () => {
  const ui = page('static/v5-demo.html')
  await flush()
  ui.context.fixture = { run_id: 'saved-run', run_token: 'fixture-demo-token' }
  ui.run('active=fixture; poll=setInterval(refresh,1500)')
  let paused = true, resolveResume
  ui.context.fetch = async (url, options) => {
    ui.calls.push({ url, options })
    if (url.endsWith('/resume')) return new Promise((done) => { resolveResume = done })
    return response({ options: { model_mode: 'live' }, status: 'running', recoverable: paused,
      followup_status: paused ? 'interrupted' : 'processing', real_model_requests: 2,
      transcript: [], trace: [] })
  }
  await ui.run('refresh()')
  assert.equal(ui.intervals.size, 0)
  assert.equal(ui.get('resume').hidden, false)
  assert.equal(ui.get('start').disabled, false)
  assert.ok(ui.get('status').textContent.includes('已安全暂停，等待手动继续'))
  assert.ok(!ui.get('status').textContent.includes('双 Agent 随访中'))
  assert.equal(ui.calls.filter((c) => c.options.method === 'POST').length, 0)
  const pending = ui.get('resume').onclick()
  await ui.get('resume').onclick()
  const posts = ui.calls.filter((c) => c.options.method === 'POST')
  assert.equal(posts.length, 1)
  assert.equal(posts[0].url, '/api/v5/simulations/saved-run/resume')
  assert.equal(posts[0].options.headers['X-Demo-Token'], 'fixture-demo-token')
  assert.equal(posts[0].options.body, undefined)
  assert.equal(ui.intervals.size, 0)
  resolveResume(response({ run_id: 'saved-run', status: 'queued' }))
  await pending
  assert.equal(ui.get('resume').hidden, true)
  assert.equal(ui.intervals.size, 1)
  paused = false
  await [...ui.intervals.values()][0].callback()
  assert.equal(ui.intervals.size, 1)
  assert.equal(ui.calls.filter((c) => c.options.method === 'POST').length, 1)
  assert.ok(!ui.calls.some((c) => c.url.includes('/messages') || c.url === '/api/chat'))
})

test('A stale resume conflict is shown without retries and a new run resets the button', async () => {
  const ui = page('static/v5-demo.html')
  await flush()
  ui.context.fixture = { run_id: 'finished-run', run_token: 'fixture-demo-token' }
  ui.run('active=fixture; $("resume").hidden=false; $("resume").disabled=false')
  ui.context.fetch = async (url, options) => {
    ui.calls.push({ url, options })
    return { ok: false, status: 409, json: async () => ({ detail: '此运行已经结束，请启动新运行' }) }
  }
  await ui.get('resume').onclick()
  assert.equal(ui.get('status').textContent, '此运行已经结束，请启动新运行')
  assert.equal(ui.intervals.size, 0)
  assert.equal(ui.calls.filter((c) => c.options.method === 'POST').length, 1)
  ui.context.fetch = async (url, options) => {
    ui.calls.push({ url, options })
    return response(url === '/api/v5/simulations' ? { run_id: 'new-run', run_token: 'new-fixture-token' } : {
      options: { model_mode: 'scripted' }, status: 'running', recoverable: false,
      followup_status: 'processing', real_model_requests: 0, transcript: [], trace: [],
    })
  }
  ui.get('scenario').value = 'stable'
  ui.get('mode').value = 'scripted'
  await ui.get('start').onclick()
  assert.equal(ui.get('resume').hidden, true)
  assert.equal(ui.get('resume').disabled, false)
  assert.equal(ui.intervals.size, 1)
})

const compiled = await transformWithOxc(readFileSync(resolve(root, 'frontend/src/App.jsx'), 'utf8'), 'App.jsx', { jsx: { runtime: 'classic' } })
function reactApp() {
  const slots = []
  let index = 0
  const calls = [], replies = []
  const context = vm.createContext({
    console,
    React: { createElement: (type, props, ...children) => ({ type, props: props || {}, children: children.flat(Infinity).filter(Boolean) }) },
    useEffect() {},
    useRef(initial) { const id = index++; if (!(id in slots)) slots[id] = { current: initial }; return slots[id] },
    useState(initial) {
      const id = index++
      if (!(id in slots)) slots[id] = initial
      return [slots[id], (value) => { slots[id] = typeof value === 'function' ? value(slots[id]) : value }]
    },
    fetch: async (url, options) => { calls.push({ url, options }); return response(await replies.shift()) },
  })
  const code = compiled.code.replace(/^import .*;\s*$/gm, '').replace('export default function App', 'function App')
  vm.runInContext(code, context)
  return { calls, replies, render() { index = 0; return vm.runInContext('App()', context) } }
}

function find(tree, predicate) {
  if (predicate(tree)) return tree
  for (const child of tree.children || []) {
    if (typeof child !== 'object') continue
    const found = find(child, predicate)
    if (found) return found
  }
}

test('React blocks completed/partial/escalated follow-ups and resets for a new one', async () => {
  for (const status of ['completed', 'partially_completed', 'escalated']) {
    const app = reactApp()
    app.replies.push({ session_id: 'react-fixture', message: '请回答当前问题。', state: { followup_status: 'waiting_for_user' } })
    let tree = app.render()
    await find(tree, (e) => e.props.className?.includes('home-start-button')).props.onClick()
    tree = app.render()
    find(tree, (e) => e.type === 'textarea').props.onChange({ target: { value: '回答' } })
    tree = app.render()
    app.replies.push({ message: '本次随访已结束。', state: { followup_status: status } })
    await find(tree, (e) => e.type === 'form').props.onSubmit({ preventDefault() {} })
    tree = app.render()
    assert.equal(find(tree, (e) => e.type === 'textarea').props.disabled, true)
    assert.equal(find(tree, (e) => e.props.className === 'send-button').props.disabled, true)
    find(tree, (e) => e.type === 'textarea').props.onChange({ target: { value: '不应再提交' } })
    await find(app.render(), (e) => e.type === 'form').props.onSubmit({ preventDefault() {} })
    assert.equal(app.calls.length, 2)
    find(tree, (e) => e.props.className === 'replay-button').props.onClick()
    app.replies.push({ session_id: 'new-fixture', message: '新随访。', state: { followup_status: 'waiting_for_user' } })
    await find(app.render(), (e) => e.props.className?.includes('home-start-button')).props.onClick()
    assert.equal(find(app.render(), (e) => e.type === 'textarea').props.disabled, false)
  }
})

test('A late React response cannot unlock a newer pending follow-up request', async () => {
  const app = reactApp()
  const startButton = (tree) => find(tree, (e) => e.props.className?.includes('home-start-button'))
  app.replies.push({ session_id: 'old-session', message: '请回答。' })
  await startButton(app.render()).props.onClick()
  find(app.render(), (e) => e.type === 'textarea').props.onChange({ target: { value: '回答' } })
  let resolveOld, resolveNew
  app.replies.push(new Promise((done) => { resolveOld = done }))
  const oldRequest = find(app.render(), (e) => e.type === 'form').props.onSubmit({ preventDefault() {} })
  find(app.render(), (e) => e.props.className === 'replay-button').props.onClick()
  app.replies.push(new Promise((done) => { resolveNew = done }))
  const newRequest = startButton(app.render()).props.onClick()
  resolveOld({ message: '旧请求结果', state: { followup_status: 'completed' } })
  await oldRequest
  await startButton(app.render()).props.onClick()
  assert.equal(app.calls.length, 3)
  resolveNew({ session_id: 'new-session', message: '新问题', state: { followup_status: 'waiting_for_user' } })
  await newRequest
  assert.equal(find(app.render(), (e) => e.type === 'textarea').props.disabled, false)
})
