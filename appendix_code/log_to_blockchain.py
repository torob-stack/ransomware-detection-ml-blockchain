from web3 import Web3
import json
import os

# === Connect to Hardhat Local Node ===
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "Could not connect to local blockchain node."

# === Load contract ABI from Hardhat build ===
with open("C:/Users/jacob/blockchain-logger/artifacts/contracts/LogLedger.sol/LogLedger.json") as f:
    artifact = json.load(f)
abi = artifact["abi"]

# === Replace with your actual deployed contract address ===
contract_address = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
contract = w3.eth.contract(address=contract_address, abi=abi)

# === Use the first account from Hardhat's local node ===
account = w3.eth.accounts[0]

# === Example Log Entry (could be dynamically created from ML alerts) ===
message = "Ransomware detected - file spike"
system_id = "ML-System-01"

# === Send Transaction ===
print(f"  Sending log to blockchain:\n  Message: {message}\n  System ID: {system_id}")
try:
    tx_hash = contract.functions.addLog(message, system_id).transact({'from': account})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Log entry recorded.")
    print("Tx Hash:", tx_receipt.transactionHash.hex())
except Exception as e:
    print("Error during transaction:", e)
