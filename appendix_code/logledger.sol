// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LogLedger {
    struct Log {
        uint256 timestamp;
        string message;
        string systemId;
    }

    Log[] private logs;

    event LogRecorded(uint256 indexed timestamp, string message, string systemId);

    function addLog(string memory message, string memory systemId) public {
        logs.push(Log(block.timestamp, message, systemId));
        emit LogRecorded(block.timestamp, message, systemId);
    }

    function getLog(uint index) public view returns (uint256, string memory, string memory) {
        require(index < logs.length, "Index out of range");
        Log memory log = logs[index];
        return (log.timestamp, log.message, log.systemId);
    }

    function getLogCount() public view returns (uint) {
        return logs.length;
    }

    function getAllLogs() public view returns (Log[] memory) {
        return logs;
    }
}
