import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'

export default function App() {
  const [productUrl, setProductUrl] = useState('')
  const [fetchStatus, setFetchStatus] = useState(null)
  const [results, setResults] = useState(null)
  const [page, setPage] = useState(0)
  const pageSize = 20 // backend’deki default limit’le eşleşmeli

  // Yeni yükleme ve aksiyon durumları
  const [loading, setLoading] = useState(false)
  const [action, setAction] = useState('') // 'fetch' veya 'analyze'
  const abortController = useRef(null)

  const isDisabled = productUrl.trim() === '' || loading

  // Duygu etiketlerine emoji
  const labelMap = {
    Very_Positive: '🌟 Çok Olumlu',
    Positive: '👍 Olumlu',
    Neutral: '😐 Nötr',
    Negative: '👎 Olumsuz',
    Very_Negative: '💥 Çok Olumsuz'
  }

  // URL normalize fonksiyonu
  function normalizeTrendyolUrl(u) {
    let url = u.trim().replace(/^view-source:/, '')
    const m = url.match(
      /https?:\/\/apigw\.trendyol\.com\/.+?\/reviews\/.+?-p-(\d+)\/.+/ 
    )
    return m ? `https://www.trendyol.com/p-${m[1]}` : url
  }

  // Page değiştiğinde analiz et (varsa)
  useEffect(() => {
    if (results && !loading) {
      handleAnalyze()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page])

  // Yorumları çek handler
  const handleFetch = async () => {
    if (isDisabled) return
    setLoading(true)
    setAction('fetch')
    abortController.current = new AbortController()
    try {
      const cleanUrl = normalizeTrendyolUrl(productUrl)
      const { data } = await axios.post(
        'http://127.0.0.1:8000/fetch_by_url',
        null,
        {
          params: { url: cleanUrl },
          signal: abortController.current.signal
        }
      )
      setFetchStatus(data)
      setResults(null)
      alert(`Fetched ${data.fetched}, Stored ${data.stored} yorum.`)
      setPage(0)
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError') {
        alert('Yorumları çekme işlemi sonlandırıldı.')
      } else {
        console.error(err)
        alert('Yorumları çekerken hata!')
      }
    } finally {
      setLoading(false)
      setAction('')
      abortController.current = null
    }
  }

  // Analiz et handler
  const handleAnalyze = async () => {
    if (isDisabled) return
    setLoading(true)
    setAction('analyze')
    abortController.current = new AbortController()
    try {
      const cleanUrl = normalizeTrendyolUrl(productUrl)
      const { data } = await axios.get(
        'http://127.0.0.1:8000/analyze_by_url',
        {
          params: {
            url: cleanUrl,
            limit: pageSize,
            offset: page * pageSize
          },
          signal: abortController.current.signal
        }
      )
      setResults(data)
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError') {
        alert('Analiz işlemi sonlandırıldı.')
      } else {
        console.error(err)
        alert('Analiz sırasında hata!')
      }
    } finally {
      setLoading(false)
      setAction('')
      abortController.current = null
    }
  }

  // İptal handler
  const handleCancel = () => {
    if (abortController.current) {
      abortController.current.abort()
    }
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-3xl mb-6 text-center">Ürün Geri Bildirim Yönetimi</h1>

      {/* URL ile Yorum Çek ve Analiz Et */}
      <div className="mb-8 border border-gray-300 p-6 rounded-lg shadow-sm bg-white">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">🛍️ URL ile Yorum Çek ve Analiz Et</h2>

        <input
          type="text"
          placeholder="Trendyol ürün URL'si"
          value={productUrl}
          onChange={e => setProductUrl(e.target.value)}
          className="border border-gray-300 px-4 py-2 rounded w-full focus:outline-none focus:ring-2 focus:ring-indigo-500 transition mb-4"
        />

      <div className="flex gap-3 flex-wrap">
        <button
          onClick={handleFetch}
          disabled={isDisabled}
          className={`
            flex-1 px-4 py-2 rounded font-semibold text-sm border transition duration-200
            bg-indigo-600 border-transparent text-gray-800
            ${action === 'fetch' ? 'opacity-80 cursor-wait' : 'hover:bg-indigo-700 cursor-pointer'}
            disabled:opacity-80 disabled:cursor-not-allowed disabled:hover:bg-indigo-600
          `}
        >
          {loading && action === 'fetch' ? 'Yorumlar çekiliyor...' : 'Yorumları Çek'}
        </button>

        <button
          onClick={handleAnalyze}
          disabled={isDisabled}
          className={`
            flex-1 px-4 py-2 rounded font-semibold text-sm border transition duration-200
            bg-indigo-600 border-transparent text-gray-800
            ${action === 'analyze' ? 'opacity-80 cursor-wait' : 'hover:bg-indigo-700 cursor-pointer'}
            disabled:opacity-80 disabled:cursor-not-allowed disabled:hover:bg-indigo-600
          `}
        >
          {loading && action === 'analyze' ? 'Analiz ediliyor...' : 'URL ile Analiz Et'}
        </button>

        {loading && (
          <button
            onClick={handleCancel}
            className="
              flex-none px-4 py-2 rounded font-semibold text-sm border
              border-red-500 bg-white text-red-500
              transition duration-200
              hover:bg-red-50 hover:border-red-600 cursor-pointer
            "
          >
            Cancel
          </button>
        )}
      </div>

        {fetchStatus && (
          <p className="mt-4 text-sm text-gray-600">
            Ürün <strong className="text-blue-600">{fetchStatus.product_id}</strong> –
            Fetched: {fetchStatus.fetched}, Stored: {fetchStatus.stored}
          </p>
        )}
      </div>

      {/* Analiz Sonuçları ve Pagination */}
      {results && (
        <div className="mb-8">
          <h2 className="text-2xl font-semibold mb-4 text-gray-800">Analiz Sonuçları</h2>

          {/* Özet Alanı */}
          <div className="bg-blue-50 border border-blue-200 p-4 rounded mb-6 text-gray-700 leading-relaxed">
            {results.summary}
          </div>

          {/* Pagination */}
          <div className="flex justify-between items-center mb-4">
            <button
              onClick={() => setPage(p => Math.max(p - 1, 0))}
              disabled={page === 0}
              className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition"
            >
              ‹ Önceki
            </button>
            <span className="text-gray-600">
              {page * pageSize + 1}–{Math.min((page + 1) * pageSize, results.total)} / {results.total}
            </span>
            <button
              onClick={() => setPage(p => p + 1)}
              disabled={(page + 1) * pageSize >= results.total}
              className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition"
            >
              Sonraki ›
            </button>
          </div>

          {/* Tablo */}
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse border border-gray-300 text-sm">
              <thead className="bg-gray-100 text-gray-700 uppercase text-xs">
                <tr>
                  <th className="border px-4 py-2 text-left">Yorum</th>
                  <th className="border px-4 py-2 text-center">Duygu</th>
                  <th className="border px-4 py-2 text-center">Yıldız</th>
                  <th className="border px-4 py-2 text-center">Tip</th>
                  <th className="border px-4 py-2 text-center">Etiketler</th>
                  <th className="border px-4 py-2 text-left w-48">Öneri</th>
                </tr>
              </thead>
              <tbody>
                {results.analyses.map((entry, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="border px-4 py-2">{entry.text}</td>
                    <td className="border px-4 py-2 text-center">{
                      labelMap[entry.sentiment.label] || entry.sentiment.label
                    }</td>
                    <td className="border px-4 py-2 text-center">{'⭐'.repeat(entry.star_rating)}</td>
                    <td className="border px-4 py-2 text-center">{entry.feedback_type}</td>
                    <td className="border px-4 py-2 text-center">{
                      entry.entities.length ?
                        entry.entities.map(e => `${e.word} (${e.entity})`).join(', ') : '—'
                    }</td>
                    <td className="border px-4 py-2">{entry.suggestion}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
