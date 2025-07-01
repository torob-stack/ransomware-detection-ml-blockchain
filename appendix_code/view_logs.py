from web3 import Web3
import json

# === Connect to Local Hardhat Node ===
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# === Replace with deployed contract address ===
contract_address = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"  # replace if different

# === Load ABI from compiled contract ===
with open("C:/Users/jacob/blockchain-logger/artifacts/contracts/LogLedger.sol/LogLedger.json") as f:
    artifact = json.load(f)
abi = artifact["abi"]

# === Connect to contract ===
contract = w3.eth.contract(address=contract_address, abi=abi)

# === Get number of logs ===
log_count = contract.functions.getLogCount().call()

print(f"\nTotal logs recorded on blockchain: {log_count}\n")

# === Fetch and display each log ===
for i in range(log_count):
    timestamp, message, system_id = contract.functions.getLog(i).call()
    from datetime import datetime
    readable_time = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"🧾 Log {i + 1}")
    print(f"  Time:     {readable_time} UTC")
    print(f"  Message:  {message}")
    print(f"  System:   {system_id}\n")
