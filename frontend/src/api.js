// Helpers with 100% field-name fidelity to gateway live schema

export async function detect(fileObj) {
  const fd = new FormData()
  fd.append("file", fileObj)
  fd.append("media_type", "image")
  // optional: strategy defaults to weighted on server; omit unless overriding

  const res = await fetch("/api/detect", {
    method: "POST",
    body: fd
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    // 422 errors are { detail: [{ loc, msg, type }] } — surface detail[0].msg
    let msg = `HTTP ${res.status}`
    if (data && Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) {
      msg = data.detail[0].msg
    } else if (data && data.detail && typeof data.detail === "string") {
      msg = data.detail
    } else if (data && data.message) {
      msg = data.message
    }
    throw new Error(msg)
  }
  return data
}

export async function getHistory({ filter = "all", search = "", limit = 50, offset = 0 }) {
  const params = new URLSearchParams()
  params.set("filter", filter)
  if (search) params.set("search", search)
  params.set("limit", String(limit))
  params.set("offset", String(offset))
  const res = await fetch(`/api/history?${params.toString()}`)
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || `History fetch failed ${res.status}`)
  return data
}

export async function getDetection(verification_id) {
  const res = await fetch(`/api/detect/${encodeURIComponent(verification_id)}`)
  const data = await res.json()
  if (!res.ok) throw new Error((data && data.detail) || `Fetch failed ${res.status}`)
  return data
}

export async function sendFeedback(feedback_token, user_verdict) {
  // expects { feedback_token, user_verdict: "correct"|"incorrect" }
  const res = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback_token, user_verdict })
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    if (data && Array.isArray(data.detail) && data.detail[0] && data.detail[0].msg) msg = data.detail[0].msg
    else if (data && data.detail && typeof data.detail === "string") msg = data.detail
    throw new Error(msg)
  }
  return data
}
