-- Dune Analytics Query: Gitcoin GR15 addresses on-chain behavior before 2022-09-22
-- Paste this in: https://dune.com/queries → New Query
-- Run time: ~2-5 minutes

WITH gitcoin_addresses AS (
  -- Replace this with actual addresses from wallets.csv
  -- This version uses a VALUES list approach
  SELECT lower(address) AS address
  FROM (VALUES
    ('0x76f69dcddd0593b0aff5fd3280c3433ddb68e0d2'),
    ('0x1bc5ebee4738fd95bd96751c45a90889f772e0f3')
    -- ... (we'll generate the full list programmatically)
  ) AS t(address)
),

-- Get transaction stats before GR15 cutoff (2022-09-22 = block ~15,579,847)
tx_stats AS (
  SELECT
    "from" AS address,
    COUNT(*) AS tx_count,
    SUM(value / 1e18) AS eth_sent,
    COUNT(DISTINCT "to") AS unique_contracts,
    MIN(block_time) AS first_tx_time,
    MAX(block_time) AS last_tx_time
  FROM ethereum.transactions
  WHERE block_time < TIMESTAMP '2022-09-22 00:00:00'
    AND "from" IN (SELECT address FROM gitcoin_addresses)
  GROUP BY "from"
),

recv_stats AS (
  SELECT
    "to" AS address,
    SUM(value / 1e18) AS eth_received
  FROM ethereum.transactions
  WHERE block_time < TIMESTAMP '2022-09-22 00:00:00'
    AND "to" IN (SELECT address FROM gitcoin_addresses)
    AND value > 0
  GROUP BY "to"
)

SELECT
  g.address,
  COALESCE(t.tx_count, 0) AS tx_count,
  COALESCE(t.eth_sent, 0) AS eth_sent,
  COALESCE(r.eth_received, 0) AS eth_received,
  COALESCE(t.eth_sent, 0) + COALESCE(r.eth_received, 0) AS total_volume,
  COALESCE(t.unique_contracts, 0) AS unique_contracts,
  CASE 
    WHEN t.first_tx_time IS NOT NULL 
    THEN DATE_DIFF('day', t.first_tx_time, TIMESTAMP '2022-09-22 00:00:00')
    ELSE NULL 
  END AS wallet_age_days,
  t.first_tx_time,
  t.last_tx_time
FROM gitcoin_addresses g
LEFT JOIN tx_stats t ON g.address = t.address
LEFT JOIN recv_stats r ON g.address = r.address
