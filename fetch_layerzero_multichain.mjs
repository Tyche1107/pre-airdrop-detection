/**
 * fetch_layerzero_multichain.mjs  v2
 * Fetch cross-chain behavior for LayerZero addresses via Etherscan V2
 * Chains: ETH, Arbitrum, Optimism, Polygon, BSC, Base
 * Global rate limiter: 9 req/s (safe under 5/s per key × 2 keys = 10/s)
 *
 * Usage: ETHERSCAN_KEYS="key1,key2" node fetch_layerzero_multichain.mjs
 */

import fs from 'fs'
import { createReadStream } from 'fs'
import { createInterface } from 'readline'

const KEYS = (process.env.ETHERSCAN_KEYS || '').split(',').filter(Boolean)
if (!KEYS.length) { console.error('Usage: ETHERSCAN_KEYS="key1,key2" node fetch_layerzero_multichain.mjs'); process.exit(1) }

const ZRO_T0    = 1718841600   // 2024-06-20 00:00:00 UTC
const ADDR_FILE = '/Users/adelinewen/Desktop/dataset/layerzero/addresses_labeled.csv'
const OUT_FILE  = '/Users/adelinewen/Desktop/dataset/layerzero/multichain_features.csv'
const PROGRESS  = '/tmp/lz_multichain_progress.log'

const CHAINS = [
  { id: 1,     name: 'eth'  },
  { id: 42161, name: 'arb'  },
  { id: 10,    name: 'op'   },
  { id: 137,   name: 'poly' },
  { id: 56,    name: 'bsc'  },
  { id: 8453,  name: 'base' },
]

// ── Global rate limiter (token bucket) ──────────────────────────────────────
const RATE_PER_SEC = 9  // conservative under 10/s limit
class RateLimiter {
  constructor(rps) {
    this.tokens = rps
    this.max    = rps
    this.queue  = []
    this.refill = setInterval(() => {
      this.tokens = Math.min(this.max, this.tokens + this.max)
      this._flush()
    }, 1000)
    // do NOT unref — we need the interval to keep the event loop alive
  }
  _flush() {
    while (this.tokens > 0 && this.queue.length > 0) {
      this.tokens--
      const { fn, resolve, reject } = this.queue.shift()
      fn().then(resolve).catch(reject)
    }
  }
  request(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject })
      this._flush()
    })
  }
}
const limiter = new RateLimiter(RATE_PER_SEC)

let keyIdx = 0
const nextKey = () => KEYS[keyIdx++ % KEYS.length]

async function fetchChainTxs(address, chainId) {
  return limiter.request(async () => {
    const key = nextKey()
    const url = `https://api.etherscan.io/v2/api?chainid=${chainId}&module=account&action=txlist` +
      `&address=${address}&startblock=0&endblock=99999999&sort=asc&apikey=${key}`
    let retries = 3
    while (retries-- > 0) {
      try {
        const res  = await fetch(url, { signal: AbortSignal.timeout(15000) })
        const data = await res.json()
        if (data.status === '1') return (data.result || []).filter(tx => parseInt(tx.timeStamp) < ZRO_T0)
        if (data.message === 'No transactions found') return []
        if (data.result === 'Max rate limit reached') {
          await new Promise(r => setTimeout(r, 2000))
          continue
        }
        return []
      } catch { await new Promise(r => setTimeout(r, 1000)) }
    }
    return []
  })
}

function computeFeatures(address, is_sybil, allChainTxs) {
  let total_tx = 0, eth_sent = 0, eth_recv = 0
  let contracts = new Set(), chains_used = 0, timestamps = []
  const per_chain = {}

  for (const { name, txs } of allChainTxs) {
    per_chain[name] = txs.length
    if (txs.length === 0) continue
    chains_used++
    total_tx += txs.length
    for (const tx of txs) {
      const val = parseFloat(tx.value) / 1e18
      if (tx.from.toLowerCase() === address) eth_sent += val
      else eth_recv += val
      if (tx.to) contracts.add(tx.to.toLowerCase())
      timestamps.push(parseInt(tx.timeStamp))
    }
  }

  const first_ts = timestamps.length ? Math.min(...timestamps) : null
  const last_ts  = timestamps.length ? Math.max(...timestamps) : null
  const wallet_age   = first_ts ? (ZRO_T0 - first_ts) / 86400 : 0
  const active_span  = (first_ts && last_ts) ? (last_ts - first_ts) / 86400 : 0
  const tx_per_day   = active_span > 0 ? total_tx / active_span : 0

  return {
    address, is_sybil,
    total_tx, eth_sent: +eth_sent.toFixed(6), eth_recv: +eth_recv.toFixed(6),
    total_volume: +(eth_sent + eth_recv).toFixed(6),
    unique_contracts: contracts.size, chains_used,
    wallet_age_days: +wallet_age.toFixed(1),
    active_span_days: +active_span.toFixed(1),
    tx_per_day: +tx_per_day.toFixed(4),
    tx_eth:  per_chain['eth']  || 0,
    tx_arb:  per_chain['arb']  || 0,
    tx_op:   per_chain['op']   || 0,
    tx_poly: per_chain['poly'] || 0,
    tx_bsc:  per_chain['bsc']  || 0,
    tx_base: per_chain['base'] || 0,
  }
}

async function loadAddresses() {
  const rows = []
  const rl = createInterface({ input: createReadStream(ADDR_FILE) })
  let first = true
  for await (const line of rl) {
    if (first) { first = false; continue }
    const parts = line.trim().split(',')
    if (parts[0]) rows.push({ address: parts[0].toLowerCase(), is_sybil: parseInt(parts[1]) || 0 })
  }
  return rows
}

function loadDone() {
  if (!fs.existsSync(OUT_FILE)) return new Set()
  const lines = fs.readFileSync(OUT_FILE, 'utf8').trim().split('\n').slice(1)
  return new Set(lines.filter(Boolean).map(l => l.split(',')[0].toLowerCase()))
}

async function main() {
  const all  = await loadAddresses()
  const done = loadDone()
  const todo = all.filter(a => !done.has(a.address))

  console.log(`Total=${all.length} Done=${done.size} Remaining=${todo.length}`)
  console.log(`Rate=${RATE_PER_SEC} req/s | ETA ~${Math.ceil(todo.length * CHAINS.length / RATE_PER_SEC / 3600)}h`)

  const HEADER = 'address,is_sybil,total_tx,eth_sent,eth_recv,total_volume,unique_contracts,chains_used,wallet_age_days,active_span_days,tx_per_day,tx_eth,tx_arb,tx_op,tx_poly,tx_bsc,tx_base'
  if (!fs.existsSync(OUT_FILE)) fs.writeFileSync(OUT_FILE, HEADER + '\n')

  const ws = fs.createWriteStream(OUT_FILE, { flags: 'a' })
  let processed = 0, t0 = Date.now()

  // Process addresses with bounded concurrency
  const ADDR_CONCURRENCY = 8  // 8 × 6 chains = 48 requests, managed by rate limiter
  for (let i = 0; i < todo.length; i += ADDR_CONCURRENCY) {
    const batch = todo.slice(i, i + ADDR_CONCURRENCY)
    await Promise.all(batch.map(async ({ address, is_sybil }) => {
      const allChainTxs = await Promise.all(
        CHAINS.map(async chain => ({ name: chain.name, txs: await fetchChainTxs(address, chain.id) }))
      )
      const feat = computeFeatures(address, is_sybil, allChainTxs)
      ws.write(Object.values(feat).join(',') + '\n')
      processed++
    }))

    if (processed % 200 === 0 || i + ADDR_CONCURRENCY >= todo.length) {
      const elapsed = (Date.now() - t0) / 1000
      const rate    = processed / elapsed
      const eta_h   = ((todo.length - processed) / rate / 3600).toFixed(1)
      const msg = `[${new Date().toISOString()}] ${done.size + processed}/${all.length} | ${rate.toFixed(1)} addr/s | ETA ${eta_h}h`
      fs.appendFileSync(PROGRESS, msg + '\n')
      process.stdout.write('\r' + msg)
    }
  }

  ws.end()
  clearInterval(limiter.refill)
  console.log(`\nDone! Output: ${OUT_FILE}`)
  process.exit(0)
}

main().catch(console.error)
