/**
 * fetch_gitcoin_erc20.mjs
 * Fetch ERC-20 token transactions for Gitcoin addresses (tokentx endpoint)
 * Gitcoin donations use DAI/USDC, not ETH — need token tx data
 * 
 * Usage: ETHERSCAN_KEYS="key1,key2" node fetch_gitcoin_erc20.mjs
 */

import fs from 'fs'

const KEYS = (process.env.ETHERSCAN_KEYS || '').split(',').filter(Boolean)
if (!KEYS.length) { console.error('Need ETHERSCAN_KEYS'); process.exit(1) }

const GR15_CUTOFF = 1663804800   // 2022-09-22
const GR15_BLOCK  = 15579847
const ADDR_FILE   = '/Users/adelinewen/Desktop/dataset/gitcoin/addresses_labeled.csv'
const OUT_FILE    = '/Users/adelinewen/Desktop/dataset/gitcoin/erc20_features.csv'
const CONCURRENCY = KEYS.length * 4
const DELAY_MS    = Math.ceil(1000 / (KEYS.length * 5))

let keyIdx = 0
const nextKey = () => KEYS[keyIdx++ % KEYS.length]
const sleep = ms => new Promise(r => setTimeout(r, ms))

// Stablecoins commonly used in Gitcoin donations
const STABLE_ADDRS = new Set([
  '0x6b175474e89094c44da98b954eedeac495271d0f', // DAI
  '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48', // USDC
  '0xdac17f958d2ee523a2206206994597c13d831ec7', // USDT
])

async function fetchTokenTx(address) {
  const key = nextKey()
  const url = `https://api.etherscan.io/v2/api?chainid=1&module=account&action=tokentx` +
    `&address=${address}&startblock=0&endblock=${GR15_BLOCK}` +
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
    return { address, token_tx_count: 0, stable_send_count: 0, stable_recv_count: 0,
             stable_volume_usd: 0, unique_token_contracts: 0,
             gitcoin_donations: 0, wallet_age_days: 0 }
  }
  const sent  = txs.filter(t => t.from.toLowerCase() === addr)
  const recv  = txs.filter(t => t.to?.toLowerCase() === addr)
  const stableSent = sent.filter(t => STABLE_ADDRS.has(t.contractAddress?.toLowerCase()))
  const stableRecv = recv.filter(t => STABLE_ADDRS.has(t.contractAddress?.toLowerCase()))
  
  const stableVolume = stableSent.reduce((s, t) => {
    const dec = parseInt(t.tokenDecimal || 18)
    return s + parseInt(t.value || 0) / Math.pow(10, dec)
  }, 0)

  // Gitcoin bulk checkout contract (common address for donations)
  const GITCOIN_CONTRACTS = new Set([
    '0x7d655c57f71464b6f83811c55d84009cd9f5221c', // bulkCheckout
    '0x69db4d94d1498b44b14b8a9f4d6c5bfc19f16945', // grants clr
  ])
  const gitcoinDonations = sent.filter(t => 
    GITCOIN_CONTRACTS.has(t.to?.toLowerCase())).length

  const uniqueContracts = new Set(txs.map(t => t.contractAddress?.toLowerCase()).filter(Boolean))
  const firstTs = parseInt(txs[0].timeStamp)
  const walletAge = Math.max(0, (GR15_CUTOFF - firstTs) / 86400)

  return {
    address,
    token_tx_count: txs.length,
    stable_send_count: stableSent.length,
    stable_recv_count: stableRecv.length,
    stable_volume_usd: stableVolume.toFixed(2),
    unique_token_contracts: uniqueContracts.size,
    gitcoin_donations: gitcoinDonations,
    wallet_age_days: walletAge.toFixed(1)
  }
}

// Load addresses
const lines = fs.readFileSync(ADDR_FILE, 'utf8').trim().split('\n').slice(1)
const addresses = lines.map(l => {
  const [addr, is_sybil, score] = l.split(',')
  return { address: addr, is_sybil: parseInt(is_sybil || 0), sad_score: parseFloat(score || 0) }
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

const header = 'address,is_sybil,sad_score,token_tx_count,stable_send_count,stable_recv_count,stable_volume_usd,unique_token_contracts,gitcoin_donations,wallet_age_days\n'
if (!fs.existsSync(OUT_FILE)) fs.writeFileSync(OUT_FILE, header)
const outStream = fs.createWriteStream(OUT_FILE, { flags: 'a' })

let done = 0
const startTime = Date.now()

for (let i = 0; i < remaining.length; i += CONCURRENCY) {
  const batch = remaining.slice(i, i + CONCURRENCY)
  await sleep(DELAY_MS)
  const results = await Promise.all(batch.map(async ({ address, is_sybil, sad_score }) => {
    const txs = await fetchTokenTx(address)
    const feat = computeFeatures(address, txs)
    return { ...feat, is_sybil, sad_score }
  }))
  for (const r of results) {
    outStream.write(`${r.address},${r.is_sybil},${r.sad_score},${r.token_tx_count},` +
      `${r.stable_send_count},${r.stable_recv_count},${r.stable_volume_usd},` +
      `${r.unique_token_contracts},${r.gitcoin_donations},${r.wallet_age_days}\n`)
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
