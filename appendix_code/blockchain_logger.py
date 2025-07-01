# blockchain_logger.py
from web3 import Web3
import json
import os

# === Set up paths ===
#artifact_path = os.path.abspath(
#    os.path.join("..", "blockchain-logger", "artifacts", "contracts", "LogLedger.sol", "LogLedger.json")
#)

# === Load ABI and contract ===
with open("C:/Users/jacob/blockchain-logger/artifacts/contracts/LogLedger.sol/LogLedger.json") as f:
    artifact = json.load(f)

abi = artifact["abi"]
contract_address = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"  # Replace if yours differs

# === Connect to local Hardhat node ===
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# === Use first default Hardhat account ===
account = w3.eth.accounts[0]

contract = w3.eth.contract(address=contract_address, abi=abi)

def record_log(message, system_id):
    print(f"Sending log to blockchain:\n  Message: {message}\n  System ID: {system_id}")

    tx_hash = contract.functions.addLog(message, system_id).transact({'from': account})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print("Log entry recorded.")
    print("Tx Hash:", tx_receipt.transactionHash.hex())
