/**
 * fetch_layerzero_onchain.mjs
 * Fetch on-chain behavior for LayerZero addresses before ZRO airdrop (2024-06-20)
 * Usage: ETHERSCAN_KEYS="key1,key2" node fetch_layerzero_onchain.mjs
 */
import fs from 'fs'

const KEYS = (process.env.ETHERSCAN_KEYS || '').split(',').filter(Boolean)
if (!KEYS.length) { console.error('Need ETHERSCAN_KEYS'); process.exit(1) }

const ZRO_T0       = 1718841600  // 2024-06-20 00:00:00 UTC
const ZRO_BLOCK    = 20130000    // approx block at 2024-06-20
const ADDR_FILE    = '/Users/adelinewen/Desktop/dataset/layerzero/addresses_labeled.csv'
const OUT_FILE     = '/Users/adelinewen/Desktop/dataset/layerzero/onchain_features.csv'
const CONCURRENCY  = KEYS.length * 4
const DELAY_MS     = Math.ceil(1000 / (KEYS.length * 5))

let keyIdx = 0
const nextKey = () => KEYS[keyIdx++ % KEYS.length]
const sleep = ms => new Promise(r => setTimeout(r, ms))

async function fetchTxList(address) {
  const key = nextKey()
  const url = `https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist` +
    `&address=${address}&startblock=0&endblock=${ZRO_BLOCK}` +
    `&sort=asc&apikey=${key}`
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(12000) })
    const data = await res.json()
    if (data.status === '1') return data.result || []
    if (data.message === 'No transactions found') return []
    return null
  } catch { return null }
}

function computeFeatures(address, txs) {
  const addr = address.toLowerCase()
  if (!txs || txs.length === 0) {
    return { address, tx_count: 0, eth_sent: 0, eth_received: 0,
             total_volume: 0, unique_contracts: 0, wallet_age_days: 0,
             tx_per_day: 0, days_active: 0 }
  }
  const sent = txs.filter(t => t.from.toLowerCase() === addr)
  const recv = txs.filter(t => t.to?.toLowerCase() === addr)
  const ethSent = sent.reduce((s, t) => s + parseInt(t.value || 0) / 1e18, 0)
  const ethRecv = recv.reduce((s, t) => s + parseInt(t.value || 0) / 1e18, 0)
  const contracts = new Set(txs.map(t => t.to?.toLowerCase()).filter(Boolean))
  const firstTs = parseInt(txs[0].timeStamp)
  const lastTs  = parseInt(txs[txs.length - 1].timeStamp)
  const walletAge = Math.max(0, (ZRO_T0 - firstTs) / 86400)
  
  // Active days (unique days with transactions)
  const activeDays = new Set(txs.map(t => Math.floor(parseInt(t.timeStamp) / 86400))).size
  const txPerDay = txs.length / Math.max(activeDays, 1)

  return {
    address,
    tx_count: txs.length,
    eth_sent: ethSent.toFixed(6),
    eth_received: ethRecv.toFixed(6),
    total_volume: (ethSent + ethRecv).toFixed(6),
    unique_contracts: contracts.size,
    wallet_age_days: walletAge.toFixed(1),
    tx_per_day: txPerDay.toFixed(3),
    days_active: activeDays
  }
}

// Load addresses
const lines = fs.readFileSync(ADDR_FILE, 'utf8').trim().split('\n').slice(1)
const addresses = lines.map(l => {
  const [addr, is_sybil] = l.split(',')
  return { address: addr, is_sybil: parseInt(is_sybil || 0) }
}).filter(r => r.address?.startsWith('0x'))

// Resume support
const done_set = new Set()
if (fs.existsSync(OUT_FILE)) {
  const ex = fs.readFileSync(OUT_FILE, 'utf8').trim().split('\n').slice(1)
  for (const l of ex) { const a = l.split(',')[0]; if (a?.startsWith('0x')) done_set.add(a.toLowerCase()) }
  console.log(`Resuming: ${done_set.size} already done`)
}
const remaining = addresses.filter(r => !done_set.has(r.address.toLowerCase()))
console.log(`Addresses: ${remaining.length} remaining | Keys: ${KEYS.length} | Concurrency: ${CONCURRENCY}`)

const header = 'address,is_sybil,tx_count,eth_sent,eth_received,total_volume,unique_contracts,wallet_age_days,tx_per_day,days_active\n'
if (!fs.existsSync(OUT_FILE)) fs.writeFileSync(OUT_FILE, header)
const outStream = fs.createWriteStream(OUT_FILE, { flags: 'a' })

let done = 0
const startTime = Date.now()

for (let i = 0; i < remaining.length; i += CONCURRENCY) {
  const batch = remaining.slice(i, i + CONCURRENCY)
  await sleep(DELAY_MS)
  const results = await Promise.all(batch.map(async ({ address, is_sybil }) => {
    const txs = await fetchTxList(address)
    const feat = computeFeatures(address, txs)
    return { ...feat, is_sybil }
  }))
  for (const r of results) {
    outStream.write(`${r.address},${r.is_sybil},${r.tx_count},${r.eth_sent},` +
      `${r.eth_received},${r.total_volume},${r.unique_contracts},` +
      `${r.wallet_age_days},${r.tx_per_day},${r.days_active}\n`)
    done++
  }
  if (done % 500 === 0 || done === remaining.length) {
    const e = (Date.now() - startTime) / 1000
    const rate = (done / e * 60).toFixed(0)
    const eta = Math.ceil((remaining.length - done) / (done / e) / 60)
    process.stdout.write(`\r[${done}/${remaining.length}] ${rate}/min | ETA ~${eta}min`)
  }
}
outStream.end()
console.log(`\nDone! → ${OUT_FILE}`)
