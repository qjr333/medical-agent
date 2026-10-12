
import { useState } from 'react'
import './App.css'

export default function App() {
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function startFollowup() {
    setLoading(true)
    setError('')
    setMessages([])

    try {
      const response = await fetch('/api/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trigger_type: 'scheduled' }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || '创建随访失败')
      }

      setSessionId(data.session_id)
      setMessages([
        {
          role: 'assistant',
          content: data.message || JSON.stringify(data, null, 2),
        },
      ])
    } catch (err) {
      setError(`无法创建随访：${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function sendMessage(event) {
    event.preventDefault()

    const message = input.trim()
    if (!message || !sessionId || loading) return

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
          session_id: sessionId,
          message,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || '发送消息失败')
      }

      setMessages((previous) => [
        ...previous,
        {
          role: 'assistant',
          content: data.message || JSON.stringify(data, null, 2),
        },
      ])
    } catch (err) {
      setError(`消息发送失败：${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">+</div>
        <div>
          <h1>健康随访助手</h1>
          <p>老年高血压主动式用药随访 Agent</p>
        </div>
        <span className="status-dot">本地演示</span>
      </header>

      <section className="intro">
        <p className="eyebrow">PATIENT FOLLOW-UP</p>
        <h2>让每一次随访，都更有连续性。</h2>
        <p>
          记录血压、了解用药情况，并通过对话完成本轮随访。
        </p>
        <button
          className="primary-button"
          onClick={startFollowup}
          disabled={loading}
        >
          {loading ? '正在创建…' : '开始新一轮随访'}
        </button>
      </section>

      <section className="chat-panel">
        <div className="panel-heading">
          <div>
            <h2>随访对话</h2>
            <p>
              {sessionId
                ? `会话已建立：${sessionId.slice(0, 8)}…`
                : '请先开始一轮随访'}
            </p>
          </div>
          <span className="online-label">● Agent</span>
        </div>

        <div className="message-list">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">♡</div>
              <h3>您好，我们开始今天的健康随访吧</h3>
              <p>点击上方按钮，创建新会话后即可开始对话。</p>
            </div>
          )}

          {messages.map((item, index) => (
            <div
              className={`message-row ${item.role}`}
              key={`${index}-${item.role}`}
            >
              <div className="avatar">
                {item.role === 'assistant' ? '健' : '我'}
              </div>
              <div className="message-bubble">{item.content}</div>
            </div>
          ))}

          {loading && sessionId && (
            <p className="loading-hint">Agent 正在处理，请稍候…</p>
          )}
        </div>

        {error && <div className="error-banner">{error}</div>}
            
        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}

            onKeyDown={(event) => {
              // Shift + Enter：保留浏览器默认行为，在文本框中换行
              if (event.key === 'Enter' && event.shiftKey) {
                return
              }

              // Enter：提交消息；中文输入法正在组词时不提交
              if (
                event.key === 'Enter' &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}

            placeholder={
              sessionId
                ? '例如：我今天早上的血压是 145/90'
                : '请先开始随访'
            }
            disabled={!sessionId || loading}
            rows={2}
          />
          <button
            className="send-button"
            type="submit"
            disabled={!sessionId || !input.trim() || loading}
          >
            发送
          </button>
        </form>
        
        <p className="disclaimer">
          本页面用于本地项目演示，不替代医生诊疗。出现明显不适时请及时寻求医疗帮助。
        </p>
      </section>
    </main>
  )
}
