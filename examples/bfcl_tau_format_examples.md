# BFCL Examples in Tau-Bench Format

This document contains representative examples of BFCL V1, V2, V3, and V4 formatted in the style of **Section C.2.2** in the Tau-Bench paper (with explicit `instruction`, target `actions` list, and structured conversation turns mapping `user`, `assistant: Function(arguments=..., name=...)`, and `tool: <response>`).

---

## 1. BFCL Version 1 (AST Single-Turn)

```json
{
  "instruction": "Find a coffee shop near me with free Wi-Fi in San Francisco.",
  "actions": [
    {
      "name": "coffee_shop.find_nearby",
      "arguments": {
        "location": "San Francisco, CA",
        "amenities": ["Wi-Fi"]
      }
    }
  ]
}
```

*   **user**: Find a coffee shop near me with free Wi-Fi in San Francisco.
*   **assistant**: Function(arguments='{"location": "San Francisco, CA", "amenities": ["Wi-Fi"]}', name='coffee_shop.find_nearby')
*   *(Note: Being single-turn, the dialogue terminates immediately upon function invocation; no tool observation is fed back to the model).*

---

## 2. BFCL Version 2 (Real-World & Live Data)

```json
{
  "instruction": "Can you retrieve the details for the user with the ID 7890, who has black as their special request?",
  "actions": [
    {
      "name": "get_user_info",
      "arguments": {
        "user_id": 7890,
        "special": "black"
      }
    }
  ]
}
```

*   **user**: Can you retrieve the details for the user with the ID 7890, who has black as their special request?
*   **assistant**: Function(arguments='{"user_id": 7890, "special": "black"}', name='get_user_info')
*   *(Note: Similar to V1, this is a single-turn tool invocation test case).*

---

## 3. BFCL Version 3 (Conversational & Multi-Turn)

```json
{
  "instruction": "You are a files and directories manager. You need to create a temporary folder in the document directory, move the final report there, perform a grep search for budget analysis, sort the file, and then compare it with the previous report to check for updates.",
  "actions": [
    {
      "name": "cd",
      "arguments": { "folder": "document" }
    },
    {
      "name": "mkdir",
      "arguments": { "dir_name": "temp" }
    },
    {
      "name": "mv",
      "arguments": { "source": "final_report.pdf", "destination": "temp" }
    },
    {
      "name": "cd",
      "arguments": { "folder": "temp" }
    },
    {
      "name": "grep",
      "arguments": { "file_name": "final_report.pdf", "pattern": "budget analysis" }
    },
    {
      "name": "sort",
      "arguments": { "file_name": "final_report.pdf" }
    },
    {
      "name": "cd",
      "arguments": { "folder": ".." }
    },
    {
      "name": "mv",
      "arguments": { "source": "previous_report.pdf", "destination": "temp" }
    },
    {
      "name": "cd",
      "arguments": { "folder": "temp" }
    },
    {
      "name": "diff",
      "arguments": { "file_name1": "final_report.pdf", "file_name2": "previous_report.pdf" }
    }
  ]
}
```

*   **user**: Move 'final_report.pdf' within document directory to 'temp' directory in document. Make sure to create the directory.
*   **assistant**: Function(arguments='{"folder":"document"}', name='cd')
*   **tool**: {"status": "success", "current_dir": "document"}
*   **assistant**: Function(arguments='{"dir_name":"temp"}', name='mkdir')
*   **tool**: {"status": "success", "msg": "Directory temp created successfully."}
*   **assistant**: Function(arguments='{"source":"final_report.pdf","destination":"temp"}', name='mv')
*   **tool**: {"status": "success", "msg": "File final_report.pdf moved to temp."}
*   **user**: Perform a detailed search using grep to identify sections in the file pertaining to 'budget analysis'.
*   **assistant**: Function(arguments='{"folder":"temp"}', name='cd')
*   **tool**: {"status": "success", "current_dir": "document/temp"}
*   **assistant**: Function(arguments='{"file_name":"final_report.pdf","pattern":"budget analysis"}', name='grep')
*   **tool**: {"status": "success", "results": ["Line 3: This is the final report content including budget analysis and other sections."]}
*   **user**: Upon identifying the requisite 'budget analysis' content, sort the 'final_report.pdf' by line for improved clarity and comprehension.
*   **assistant**: Function(arguments='{"file_name":"final_report.pdf"}', name='sort')
*   **tool**: {"status": "success", "msg": "File final_report.pdf sorted."}
*   **user**: Move 'previous_report.pdf' in document directory to temp as well and having final report also there, proceed to juxtapose it with 'previous_report.pdf' to detect any critical alterations.
*   **assistant**: Function(arguments='{"folder":".."}', name='cd')
*   **tool**: {"status": "success", "current_dir": "document"}
*   **assistant**: Function(arguments='{"source":"previous_report.pdf","destination":"temp"}', name='mv')
*   **tool**: {"status": "success", "msg": "File previous_report.pdf moved to temp."}
*   **assistant**: Function(arguments='{"folder":"temp"}', name='cd')
*   **tool**: {"status": "success", "current_dir": "document/temp"}
*   **assistant**: Function(arguments='{"file_name1":"final_report.pdf","file_name2":"previous_report.pdf"}', name='diff')
*   **tool**: {"status": "success", "differences": ["- Year2024 This is the final report...", "+ Year203 This is the previous report..."]}

---

## 4. BFCL Version 4 (Agentic & Multi-Hop Web Search)

```json
{
  "instruction": "Some countries are known for producing luxury goods, including the world's most expensive tea. In April 2025, who is the richest billionaire (according to Forbes) from the country that produces the most expensive tea?",
  "actions": [
    {
      "name": "search_engine_query",
      "arguments": { "keywords": "world's most expensive tea" }
    },
    {
      "name": "search_engine_query",
      "arguments": { "keywords": "richest billionaire from China Forbes April 2025" }
    }
  ]
}
```

*   **user**: Some countries are known for producing luxury goods, including the world's most expensive tea. In April 2025, who is the richest billionaire (according to Forbes) from the country that produces the most expensive tea?
*   **assistant**: Function(arguments='{"keywords": "world\'s most expensive tea"}', name='search_engine_query')
*   **tool**: "Da Hong Pao is the legendary and world's most expensive tea, costing over $1.2 million per kilogram. It is produced in the Wuyi Mountains of Fujian Province, China."
*   **assistant**: Function(arguments='{"keywords": "richest billionaire from China Forbes April 2025"}', name='search_engine_query')
*   **tool**: "According to the Forbes Billionaires list from April 2025, Zhong Shanshan / Zhang Yiming represents the top tier, with Zhang Yiming (ByteDance founder) emerging as the wealthiest billionaire in China at that time."
*   **assistant**: In April 2025, the richest billionaire from China—the country that produces the world's most expensive tea (Da Hong Pao)—is Zhang Yiming.
