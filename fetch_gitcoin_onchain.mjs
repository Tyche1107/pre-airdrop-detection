/**
 * fetch_gitcoin_onchain.mjs
 * Fetch on-chain behavior for Gitcoin GR15 addresses before 2022-09-22
 * Multi-key parallel scraper (register multiple free Etherscan accounts)
 * 
 * Usage:
 *   ETHERSCAN_KEYS="key1,key2,key3" node fetch_gitcoin_onchain.mjs
 */

import fs from 'fs'
import { createReadStream } from 'fs'
import { createInterface } from 'readline'
import path from 'path'

const KEYS = (process.env.ETHERSCAN_KEYS || '').split(',').filter(Boolean)
if (!KEYS.length) {
  console.error('Usage: ETHERSCAN_KEYS="key1,key2,key3" node fetch_gitcoin_onchain.mjs')
  process.exit(1)
}

const GR15_CUTOFF  = 1663804800  // 2022-09-22 00:00:00 UTC
const GR15_BLOCK   = 15579847    // approx block at 2022-09-22
const ADDR_FILE    = '/Users/adelinewen/Desktop/dataset/gitcoin/addresses_labeled.csv'
const OUT_FILE     = '/Users/adelinewen/Desktop/dataset/gitcoin/onchain_features.csv'
const CONCURRENCY  = KEYS.length * 4  // 4 parallel per key
const DELAY_MS     = Math.ceil(1000 / (KEYS.length * 4))  // respect rate limit

let keyIdx = 0
const nextKey = () => KEYS[keyIdx++ % KEYS.length]
const sleep = ms => new Promise(r => setTimeout(r, ms))

async function fetchTxList(address) {
  const key = nextKey()
  const url = `https://api.etherscan.io/v2/api?chainid=1&module=account&action=txlist` +
    `&address=${address}&startblock=0&endblock=${GR15_BLOCK}` +
    `&sort=asc&apikey=${key}`
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(10000) })
    const data = await res.json()
    if (data.status === '1') return data.result || []
    if (data.message === 'No transactions found') return []
    return null  // error
  } catch { return null }
}

function computeFeatures(address, txs) {
  if (!txs || txs.length === 0) {
    return { address, tx_count: 0, eth_sent: 0, eth_received: 0,
             total_volume: 0, unique_contracts: 0, wallet_age_days: 0,
             first_tx_ts: null, last_tx_ts: null }
  }
  const sent = txs.filter(t => t.from.toLowerCase() === address.toLowerCase())
  const recv = txs.filter(t => t.to?.toLowerCase() === address.toLowerCase())
  const ethSent = sent.reduce((s, t) => s + parseInt(t.value || 0) / 1e18, 0)
  const ethRecv = recv.reduce((s, t) => s + parseInt(t.value || 0) / 1e18, 0)
  const contracts = new Set(txs.map(t => t.to?.toLowerCase()).filter(Boolean))
  const firstTs = parseInt(txs[0].timeStamp)
  const lastTs  = parseInt(txs[txs.length - 1].timeStamp)
  const walletAge = (GR15_CUTOFF - firstTs) / 86400
  return {
    address, tx_count: txs.length,
    eth_sent: ethSent.toFixed(6), eth_received: ethRecv.toFixed(6),
    total_volume: (ethSent + ethRecv).toFixed(6),
    unique_contracts: contracts.size,
    wallet_age_days: Math.max(0, walletAge).toFixed(1),
    first_tx_ts: firstTs, last_tx_ts: lastTs
  }
}

// Load addresses
const lines = fs.readFileSync(ADDR_FILE, 'utf8').trim().split('\n').slice(1)
const addresses = lines.map(l => {
  const [addr, is_sybil, score] = l.split(',')
  return { address: addr, is_sybil: parseInt(is_sybil || 0), sad_score: parseFloat(score || 0) }
}).filter(r => r.address?.startsWith('0x'))

// Load already-done addresses for resume
const doneAddresses = new Set()
if (fs.existsSync(OUT_FILE)) {
  const existing = fs.readFileSync(OUT_FILE, 'utf8').trim().split('\n').slice(1)
  for (const line of existing) {
    const addr = line.split(',')[0]
    if (addr?.startsWith('0x')) doneAddresses.add(addr.toLowerCase())
  }
  console.log(`Resuming: ${doneAddresses.size} already done`)
}

const remaining = addresses.filter(r => !doneAddresses.has(r.address.toLowerCase()))
console.log(`Addresses: ${remaining.length} remaining | Keys: ${KEYS.length} | Concurrency: ${CONCURRENCY}`)
console.log(`Output: ${OUT_FILE}`)

// Append mode (keep existing data)
const header = 'address,is_sybil,sad_score,tx_count,eth_sent,eth_received,total_volume,unique_contracts,wallet_age_days,first_tx_ts,last_tx_ts\n'
if (!fs.existsSync(OUT_FILE)) fs.writeFileSync(OUT_FILE, header)
const outStream = fs.createWriteStream(OUT_FILE, { flags: 'a' })

let done = 0, errors = 0
const startTime = Date.now()

// Process in batches
for (let i = 0; i < remaining.length; i += CONCURRENCY) {
  const batch = remaining.slice(i, i + CONCURRENCY)
  const results = await Promise.all(batch.map(async ({ address, is_sybil, sad_score }) => {
    await sleep(DELAY_MS)
    const txs = await fetchTxList(address)
    const feat = computeFeatures(address, txs)
    return { ...feat, is_sybil, sad_score }
  }))

  for (const r of results) {
    outStream.write(
      `${r.address},${r.is_sybil},${r.sad_score},${r.tx_count},${r.eth_sent},` +
      `${r.eth_received},${r.total_volume},${r.unique_contracts},` +
      `${r.wallet_age_days},${r.first_tx_ts || ''},${r.last_tx_ts || ''}\n`
    )
    done++
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0)
  const rate = (done / elapsed * 60).toFixed(0)
  const eta = Math.ceil((addresses.length - done) / (done / elapsed) / 60)
  if (done % 500 === 0 || done === remaining.length) {
    process.stdout.write(`\r[${done}/${remaining.length}] ${rate}/min | ETA ~${eta}min`)
  }
}

outStream.end()
console.log(`\nDone! ${done} addresses → ${OUT_FILE}`)
