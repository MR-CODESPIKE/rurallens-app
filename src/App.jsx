import { useEffect, useMemo, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web'
import diseaseData from '../data/diseases.json'
import ragIndex from '../rag/index.json'

const domainOrder = ['crop', 'animal', 'human']
const domainIcons = { crop: '◈', animal: '◇', human: '○' }
const availabilityLabel = { good: 'Good data', partial: 'Partial data', none: 'No open image set' }
const trainedModels = {
  crop: { ids: ['maize_gray_leaf_spot', 'potato_late_blight', 'tomato_late_blight'], labels: ['Maize gray leaf spot', 'Potato late blight', 'Tomato late blight'] },
  animal: { ids: ['lumpy_skin_disease', 'normal_animal_skin'], labels: ['Lumpy skin disease', 'Normal animal skin'] },
  human: { ids: ['acne_vulgaris', 'vitiligo'], labels: ['Acne vulgaris', 'Vitiligo'] },
}
const sessionCache = new Map()

function softmax(values) {
  const max = Math.max(...values)
  const exps = values.map((value) => Math.exp(value - max))
  const total = exps.reduce((sum, value) => sum + value, 0)
  return exps.map((value) => value / total)
}

function labelFor(domain, id, index) {
  const configured = trainedModels[domain]
  const source = diseaseData.domains[domain].classes.find((item) => item.id === id)
  return source || { id, name: configured.labels[index] || id.replaceAll('_', ' '), availability: 'partial', summary: 'This trained subset class requires local expert confirmation.' }
}

async function loadVisionSession(domain) {
  if (!sessionCache.has(domain)) {
    const promise = ort.InferenceSession.create(`/models/${domain}-mobilenet_v3_small.onnx`, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
    sessionCache.set(domain, promise)
  }
  return sessionCache.get(domain)
}

const RAG_TOKEN_RE = /[a-z0-9]+/g
const RAG_STOPWORDS = new Set(['the', 'and', 'are', 'from', 'this', 'that', 'with', 'what', 'should', 'look', 'next', 'use', 'for'])

function ragTokens(text) {
  return (text.toLowerCase().match(RAG_TOKEN_RE) || []).filter((token) => token.length > 2 && !RAG_STOPWORDS.has(token))
}

function retrieve(domain, prediction, question = '') {
  const query = `${prediction} ${question}`
  const queryTokens = ragTokens(query)
  const querySet = new Set(queryTokens)
  const scores = new Map()
  for (const token of querySet) {
    const idf = ragIndex.idf[token] || 1
    for (const [recordIndex, termFrequency] of ragIndex.postings[token] || []) {
      const record = ragIndex.records[recordIndex]
      if (!record || (record.domain !== domain && record.domain !== 'general' && record.domain !== null)) continue
      const contribution = (1 + Math.log1p(termFrequency)) * idf
      scores.set(recordIndex, (scores.get(recordIndex) || 0) + contribution)
    }
  }
  const phrase = query.toLowerCase().replace(/[_-]+/g, ' ').trim()
  return [...scores.entries()].map(([recordIndex, score]) => {
    const record = ragIndex.records[recordIndex]
    const normalizedText = record.text.toLowerCase().replace(/[_-]+/g, ' ')
    const exactBoost = phrase && normalizedText.includes(phrase) ? 2 : 0
    const classBoost = record.class_id && querySet.has(record.class_id) ? 3 : 0
    return { ...record, score: score + exactBoost + classBoost }
  }).sort((a, b) => b.score - a.score).slice(0, 3)
}

async function imageTensor(file) {
  const url = URL.createObjectURL(file)
  try {
    const image = new Image()
    image.src = url
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject })
    const canvas = document.createElement('canvas')
    canvas.width = 224; canvas.height = 224
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(image, 0, 0, 224, 224)
    const pixels = ctx.getImageData(0, 0, 224, 224).data
    const mean = [0.485, 0.456, 0.406]; const std = [0.229, 0.224, 0.225]
    const data = new Float32Array(3 * 224 * 224)
    for (let y = 0; y < 224; y += 1) for (let x = 0; x < 224; x += 1) {
      const p = (y * 224 + x) * 4; const index = y * 224 + x
      data[index] = (pixels[p] / 255 - mean[0]) / std[0]
      data[224 * 224 + index] = (pixels[p + 1] / 255 - mean[1]) / std[1]
      data[2 * 224 * 224 + index] = (pixels[p + 2] / 255 - mean[2]) / std[2]
    }
    return new ort.Tensor('float32', data, [1, 3, 224, 224])
  } finally { URL.revokeObjectURL(url) }
}

function App() {
  const [domain, setDomain] = useState('crop')
  const [imageUrl, setImageUrl] = useState('')
  const [fileName, setFileName] = useState('')
  const [results, setResults] = useState([])
  const [sources, setSources] = useState([])
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('Ready offline')
  const [question, setQuestion] = useState('')
  const [answerSources, setAnswerSources] = useState([])
  const [runtimeError, setRuntimeError] = useState('')
  const fileRef = useRef(null)
  const currentDomain = diseaseData.domains[domain]
  const classCount = useMemo(() => currentDomain.classes.length, [currentDomain])
  const coverage = trainedModels[domain].ids.length

  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl) }, [imageUrl])

  async function handleFile(file) {
    if (!file) return
    setBusy(true); setRuntimeError(''); setStatus('Loading local vision model…'); setFileName(file.name)
    const nextUrl = URL.createObjectURL(file); setImageUrl(nextUrl)
    try {
      const session = await loadVisionSession(domain)
      setStatus('Running vision model on device…')
      const tensor = await imageTensor(file)
      const output = await session.run({ [session.inputNames[0]]: tensor })
      const logits = output[session.outputNames[0]].data
      const probabilities = softmax(Array.from(logits))
      const matches = probabilities.map((confidence, index) => ({ ...labelFor(domain, trainedModels[domain].ids[index], index), confidence }))
        .sort((a, b) => b.confidence - a.confidence).slice(0, 3)
      const retrieved = retrieve(domain, `${matches[0]?.id || ''} ${matches[0]?.name || domain}`)
      setResults(matches); setSources(retrieved); setAnswerSources([]); setStatus('Offline triage complete')
    } catch (error) {
      setResults([]); setSources([]); setRuntimeError(`Local model unavailable: ${error.message}`); setStatus('Offline model error')
    } finally { setBusy(false) }
  }

  function chooseDomain(next) {
    setDomain(next); setResults([]); setSources([]); setAnswerSources([]); setRuntimeError(''); setImageUrl(''); setFileName(''); setStatus('Ready offline')
  }

  function answerQuestion() {
    if (!question.trim() || !results[0]) return
    setAnswerSources(retrieve(domain, `${results[0].id} ${results[0].name}`, question))
  }

  const best = results[0]
  return (
    <div className="app-shell">
      <header className="topbar"><div className="brand-mark">RL</div><div><p className="eyebrow">OFFLINE FIELD TOOL</p><h1>RuralLens</h1></div><div className="offline-pill"><span className="status-dot" /> {status}</div></header>
      <main className="content">
        <section className="hero"><div><p className="eyebrow accent">AFRICA-FIRST VISUAL TRIAGE</p><h2>See a possible problem. Get a grounded next step.</h2><p className="hero-copy">The selected MobileNetV3 model, local retrieval corpus, and explanation contract run on this device. No image or question leaves the device.</p></div><div className="hero-stat"><strong>60</strong><span>focused classes</span></div></section>
        <section className="panel"><div className="section-heading"><div><p className="eyebrow">01 / CHOOSE DOMAIN</p><h3>What are you checking?</h3></div><span className="class-count">{classCount} classes · {coverage} trained</span></div><div className="domain-grid">{domainOrder.map((item) => <button key={item} className={`domain-card ${domain === item ? 'selected' : ''}`} onClick={() => chooseDomain(item)}><span className="domain-icon">{domainIcons[item]}</span><span><strong>{diseaseData.domains[item].display_name}</strong><small>{trainedModels[item].ids.length} trained targets · {diseaseData.domains[item].classes.filter((x) => x.availability !== 'none').length} data-ready targets</small></span></button>)}</div>{domain === 'human' && <div className="notice human-disclaimer"><span className="notice-mark">!</span><div><strong>Human-skin disclaimer</strong><p>Human-skin results are not a medical diagnosis. This offline tool is for visual triage only; consult a qualified health professional.</p></div></div>}</section>
        <section className="panel capture-panel"><div className="section-heading"><div><p className="eyebrow">02 / ADD PHOTO</p><h3>Use a clear, well-lit image</h3></div><span className="local-badge">LOCAL ONLY</span></div><input ref={fileRef} type="file" accept="image/*" capture="environment" className="sr-only" onChange={(e) => handleFile(e.target.files?.[0])} />{!imageUrl ? <button className="upload-zone" onClick={() => fileRef.current?.click()}><span className="upload-icon">+</span><strong>Choose photo or open camera</strong><span>JPG, PNG · processed on this device</span></button> : <div className="photo-review"><img src={imageUrl} alt="Selected field image" /><div className="photo-meta"><strong>{fileName}</strong><span>{busy ? 'Analyzing locally…' : 'Image stays on this device'}</span><button className="text-button" onClick={() => fileRef.current?.click()}>Choose another</button></div></div>}</section>
        {runtimeError && <section className="notice"><span className="notice-mark">!</span><div><strong>Model coverage notice</strong><p>{runtimeError} This domain currently exposes only the trained subset shown above; unsupported classes are not silently guessed.</p></div></section>}
        {best && <section className="result-stack" aria-live="polite"><div className="result-header"><div><p className="eyebrow accent">03 / LOCAL RESULT</p><h3>Possible visual matches</h3></div><span className="model-chip">MOBILENETV3 · ON DEVICE</span></div><div className="primary-result"><div><span className="confidence-label">TOP RANK · PROVISIONAL</span><h4>{best.name}</h4><p>This is a visual screening signal, not a confirmed diagnosis. Compare the image with the signs below and seek local expert confirmation when uncertain.</p></div><div className="confidence-ring"><strong>{Math.round(best.confidence * 100)}%</strong><span>score</span></div></div><div className="match-list">{results.map((item, index) => <div className="match-row" key={item.id}><span className="rank">0{index + 1}</span><span className="match-name">{item.name}<small>{availabilityLabel[item.availability] || 'Trained subset'}</small></span><strong>{Math.round(item.confidence * 100)}%</strong></div>)}</div><div className="explanation-card"><div className="explanation-heading"><span className="explanation-icon">i</span><div><p className="eyebrow">LOCAL KNOWLEDGE BASE</p><h4>What to do next</h4></div></div><p>{currentDomain.disclaimer}</p><div className="source-list">{sources.map((source) => <div className="source-item" key={source.id}><span>VERIFIED</span><p>{source.text}</p><a href={source.url} target="_blank" rel="noreferrer">{source.org} source</a></div>)}</div></div><div className="question-card"><label htmlFor="question">Ask a follow-up question</label><div className="question-row"><input id="question" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && answerQuestion()} placeholder="e.g. What should I look for next?" /><button onClick={answerQuestion}>Answer locally</button></div><small>Retrieval uses only verified local snippets. A full local language-model adapter is reported separately until its forward pass is validated on-device.</small>{answerSources.length > 0 && <div className="source-list answer-list">{answerSources.map((source) => <div className="source-item" key={`answer-${source.id}`}><span>LOCAL CONTEXT</span><p>{source.text}</p></div>)}</div>}</div></section>}
        <section className="notice"><span className="notice-mark">!</span><div><strong>Use with care</strong><p>RuralLens is an educational triage aid. It cannot confirm a disease, toxin, or treatment. Human-skin results are not medical diagnoses; consult a qualified professional.</p></div></section>
      </main><footer><span>RuralLens · offline by design</span><span>Model contract: vision → retrieval → explanation</span></footer>
    </div>
  )
}

export default App
