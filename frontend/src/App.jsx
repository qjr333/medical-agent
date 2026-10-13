
import { useEffect, useRef, useState } from 'react'
import './App.css'

const FOLLOWUP_SCENARIOS = [
  {
    triggerType: 'scheduled',
    title: '定时随访',
    description: '到达每日计划健康随访时间，系统主动发起本次健康随访。',
    icon: '◷',
    theme: 'scheduled',
    shortDescription: '每日计划时间已到',
  },
  {
    triggerType: 'high_bp',
    title: '血压异常',
    description: '系统模拟收到异常血压记录，主动联系患者核实健康状况。',
    icon: '♥',
    theme: 'high-bp',
    shortDescription: '检测到异常血压记录',
  },
  {
    triggerType: 'missed_measurement',
    title: '连续未测血压',
    description: '系统模拟近期连续未获得新的家庭血压记录，主动提醒患者测量。',
    icon: '◉',
    theme: 'missed',
    shortDescription: '近期缺少血压测量记录',
  },
]

export default function App() {
  const [page, setPage] = useState('home')

  // 下一次点击首页按钮时使用的场景序号。
  const [scenarioIndex, setScenarioIndex] = useState(0)

  // 当前对话实际使用的场景序号，与下一轮序号分开保存。
  const [activeScenarioIndex, setActiveScenarioIndex] = useState(0)

  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const messageListRef = useRef(null)
  const requestLockRef = useRef(false)
  const pageRef = useRef('home')
  const requestVersionRef = useRef(0)

  const currentScenario = FOLLOWUP_SCENARIOS[activeScenarioIndex]

  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop =
        messageListRef.current.scrollHeight
    }
  }, [messages, loading])

  async function readResponse(response) {
    const text = await response.text()

    let data
    try {
      data = text ? JSON.parse(text) : {}
    } catch {
      data = { detail: text || '服务器返回了无法解析的内容' }
    }

    if (!response.ok) {
      throw new Error(
        data.detail ||
          data.message ||
          `请求失败（HTTP ${response.status}）`,
      )
    }

    return data
  }

  async function startFollowup() {
    if (requestLockRef.current) return

    requestLockRef.current = true
    const requestVersion = ++requestVersionRef.current

    setLoading(true)
    setError('')
    setInput('')
    setSessionId('')
    setMessages([])

    const selectedIndex = scenarioIndex
    const selectedScenario = FOLLOWUP_SCENARIOS[selectedIndex]

    try {
      const response = await fetch('/api/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trigger_type: selectedScenario.triggerType,
        }),
      })

      const data = await readResponse(response)

      // 如果用户已经离开本次页面，不再切回对话页。
      if (requestVersion !== requestVersionRef.current) return

      if (!data.session_id) {
        throw new Error('服务器没有返回有效的会话 ID，请检查后端接口。')
      }

      setActiveScenarioIndex(selectedIndex)
      setSessionId(data.session_id)
      setMessages([
        {
          role: 'assistant',
          content: data.message || '您好，我们开始本次健康随访吧。',
        },
      ])

      setScenarioIndex(
        (selectedIndex + 1) % FOLLOWUP_SCENARIOS.length,
      )

      pageRef.current = 'chat'
      setPage('chat')
    } catch (err) {
      if (requestVersion === requestVersionRef.current) {
        setError(`创建主动随访失败：${err.message}`)
      }
    } finally {
      if (requestVersion === requestVersionRef.current) {
        setLoading(false)
      }
      requestLockRef.current = false
    }
  }

  async function sendMessage(event) {
    event.preventDefault()

    const message = input.trim()

    if (!message || !sessionId || requestLockRef.current) return

    requestLockRef.current = true
    const requestVersion = ++requestVersionRef.current
    const currentSessionId = sessionId

    setMessages((previous) => [
      ...previous,
      { role: 'user', content: message },
    ])
    setInput('')
    setLoading(true)
    setError('')

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          message,
        }),
      })

      const data = await readResponse(response)

      if (
        requestVersion !== requestVersionRef.current ||
        pageRef.current !== 'chat'
      ) {
        return
      }

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: data.message || '本轮处理已完成。',
        },
      ])
    } catch (err) {
      if (
        requestVersion === requestVersionRef.current &&
        pageRef.current === 'chat'
      ) {
        setError(`消息发送失败：${err.message}`)
      }
    } finally {
      if (requestVersion === requestVersionRef.current) {
        setLoading(false)
      }
      requestLockRef.current = false
    }
  }

  function returnHome() {
    // 使正在进行的请求失效，避免返回首页后被旧请求切回对话页。
    requestVersionRef.current += 1
    requestLockRef.current = false
    pageRef.current = 'home'

    setPage('home')
    setSessionId('')
    setMessages([])
    setInput('')
    setError('')
    setLoading(false)

    // 注意：不重置 scenarioIndex。
    // 下一次点击首页按钮将进入下一轮触发场景。
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">+</div>

        <div className="brand-copy">
          <h1>银龄守护</h1>
          <p>老年高血压主动式用药随访 Agent</p>
        </div>

        <span className="status-dot">
          <span className="status-dot-inner" />
          本地演示
        </span>
      </header>

      {page === 'home' ? (
        <section className="home-page">
          <div className="hero-section">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              PROACTIVE HEALTHCARE
            </div>

            <h2>
              让健康管理
              <br />
              <span>主动一步。</span>
            </h2>

            <p className="hero-description">
              面向老年高血压患者的主动随访助手。
              从定时随访、血压异常到连续未测血压，
              通过模拟不同触发场景，体验完整的主动健康管理流程。
            </p>

            <button
              type="button"
              className="primary-button home-start-button"
              onClick={startFollowup}
              disabled={loading}
            >
              <span>
                {loading ? '正在创建随访…' : '模拟触发主动随访'}
              </span>
              {!loading && <span className="button-arrow">→</span>}
            </button>

            {error && (
              <div className="error-banner home-error" role="alert">
                {error}
              </div>
            )}

            <div className="hero-footnote">
              <span className="pulse-dot" />
              当前为模拟演示模式，点击按钮即可触发下一轮场景
            </div>
          </div>

          <section className="scenario-section">
            <div className="section-heading">
              <div>
                <p className="section-kicker">DEMO SCENARIOS</p>
                <h3>三种主动随访触发场景</h3>
              </div>
              <span className="scenario-count">03 SCENARIOS</span>
            </div>

            <div className="scenario-grid">
              {FOLLOWUP_SCENARIOS.map((scenario, index) => (
                <article
                  className={`scenario-card ${scenario.theme}`}
                  key={scenario.triggerType}
                >
                  <div className="scenario-card-top">
                    <span className="scenario-icon" aria-hidden="true">
                      {scenario.icon}
                    </span>
                    <span className="scenario-number">
                      0{index + 1}
                    </span>
                  </div>

                  <h4>{scenario.title}</h4>
                  <p>{scenario.description}</p>

                  <div className="scenario-card-bottom">
                    <span className="scenario-indicator" />
                    <span>{scenario.shortDescription}</span>
                  </div>
                </article>
              ))}
            </div>

            <div className="demo-tip">
              <span className="demo-tip-icon">i</span>
              <p>
                演示顺序为定时随访 → 血压异常 → 连续未测血压，
                三轮循环。退出对话后，下一次点击将进入下一种场景。
              </p>
            </div>
          </section>
        </section>
      ) : (
        <section className="chat-page">
          <div className={`trigger-banner ${currentScenario.theme}`}>
            <div className="trigger-banner-icon" aria-hidden="true">
              {currentScenario.icon}
            </div>

            <div className="trigger-banner-copy">
              <span className="trigger-eyebrow">
                PROACTIVE FOLLOW-UP · 主动触发
              </span>
              <h2>{currentScenario.title}</h2>
              <p>{currentScenario.description}</p>
            </div>

            <span className="trigger-step">
              第 {activeScenarioIndex + 1} 轮
            </span>
          </div>

          <section className="chat-panel">
            <div className="panel-heading">
              <div className="panel-title-group">
                <div className="agent-avatar">健</div>
                <div>
                  <h3>健康随访对话</h3>
                  <p>
                    {sessionId
                      ? `会话 ID：${sessionId.slice(0, 8)}…`
                      : '正在建立会话…'}
                  </p>
                </div>
              </div>

              <span className="online-label">
                <span className="pulse-dot" />
                Agent 在线
              </span>
            </div>

            <div className="message-list" ref={messageListRef}>
              {messages.map((item, index) => (
                <div
                  className={`message-row ${item.role}`}
                  key={`${sessionId}-${index}-${item.role}`}
                >
                  <div className="avatar">
                    {item.role === 'assistant' ? '健' : '我'}
                  </div>

                  <div className="message-content">
                    <span className="message-sender">
                      {item.role === 'assistant' ? '健康随访助手' : '我'}
                    </span>
                    <div className="message-bubble">{item.content}</div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="typing-indicator">
                  <span />
                  <span />
                  <span />
                  <p>健康随访助手正在思考…</p>
                </div>
              )}
            </div>

            {error && (
              <div className="error-banner" role="alert">
                {error}
              </div>
            )}

            <form className="composer" onSubmit={sendMessage}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && event.shiftKey) return

                  if (
                    event.key === 'Enter' &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault()
                    event.currentTarget.form?.requestSubmit()
                  }
                }}
                placeholder="请输入您的血压、服药情况或身体感受…"
                disabled={!sessionId || loading}
                rows={2}
              />

              <button
                className="send-button"
                type="submit"
                disabled={!sessionId || !input.trim() || loading}
              >
                <span>发送</span>
                <span aria-hidden="true">↑</span>
              </button>
            </form>

            <div className="chat-bottom">
              <p className="disclaimer">
                本页面用于项目演示，不替代医生诊疗。出现明显不适时请及时寻求医疗帮助。
              </p>

              <button
                type="button"
                className="replay-button"
                onClick={returnHome}
              >
                <span aria-hidden="true">↶</span>
                重新演示
              </button>
            </div>
          </section>
        </section>
      )}

      <footer className="app-footer">
        <span>银龄守护 · 主动健康管理</span>
        <span>PROACTIVE CARE FOR HEALTHIER AGING</span>
      </footer>
    </main>
  )
}
