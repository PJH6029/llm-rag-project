from langchain.prompts import ChatPromptTemplate

generate_questions_prompt = ChatPromptTemplate.from_template(
"""Create {num_questions} question-answer pairs using the context provided below.

Context consists of the two kinds of context: base context and additional context.
Base context is the most base information and additional context contains information about changes or updates to the base context,
which are made according to the version updates of the documents or by the request of the specific clients.

You should create questions based on the context, and answer them concisely.
Questions can ask about the knowledge in the context, the changes in the context, or even the unrelated information to the context.
If you ask questions about the unrelated information, the answer should be that the context does not contain the information about the question.

You should answer step by step.
You should first find the answer from the base context,
and then consider the additional context to see if there are any updates or changes to the answer.
If there are several changes, you should consider the most recent additional context to provide the most up-to-date answer,
also comparing the changes to the base context.

You should respond only question-answer pairs, without any additional information.
Each pair is parenthesized with <pair></pair> tags, and separated by a newline.
In one pair, the question and the answer are parenthesized with <question></question> and <answer></answer> tags, respectively.

You can use markdown to format your answers.
--------------------------------------------------
**** Example ****
<context>
    <base-context>
    --- Document: Datacenter NVMe SSD Specification v2.0r21.pdf ---
    CHUNK 0
    SIMILARITY_SCORE= 0.99
    TEXT=
    16.1 NVMe CLI Management Utility
    The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli) shall be used as one of the management utilities for NVMe devices.
    Requirement ID: UTIL-1
    Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - Log page reads including vendor log pages.
    - SMART status.
    METADATA=
    {{
        "document_metadata": {{
            "doc_name": "Datacenter NVMe SSD Specification v2.0r21.pdf",
            "doc_type": "base",
            "version": "v2.0r21",
            "uri": "https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf"
        }},
        "chunk_metadata": {{
            "score": HIGH,
            "excerpt_page_number": "110",
        }}
    }}

    --- Document: datacenter-nvme-ssd-specification-v2-5-pdf.pdf ---
    CHUNK 0
    SIMILARITY SCORE= 0.97
    TEXT=
    18.1 NVMe CLI Management Utility
    The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli/tree/master/plugins/ocp) shall be used as one of the management utilities for NVMe devices.
    Requirement ID: UTIL-1
    Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - Log page reads including vendor log pages.
    - SMART status.
    METADATA=
    {{
        "document_metadata": {{
            "doc_name": "datacenter-nvme-ssd-specification-v2-5-pdf.pdf",
            "doc_type": "base",
            "version": "v2.5",
            "uri": "https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf"
        }},
        "chunk_metadata": {{
            "score": HIGH,
            "excerpt_page_number": "170",
        }}
    }}

    </base-context>
    <additional-context>
    --- Document: Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf ---
    CHUNK 0
    SIMILARITY SCORE= 0.89
    TEXT=
    2. Change List
    2.1. Utilization Features
    Requirement ID: UTIL-1
    Description: The list of commands to be tested has been updated.
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - SMART status.
    METADATA=
    {{
        "document_metadata": {{
            "doc_name": "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf",
            "doc_type": "additional",
            "version": "v2.5-addendum-v0.20",
            "uri": "https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf"
        }},
        "chunk_metadata": {{
            "score": HIGH,
            "excerpt_page_number": "3",
            "base_doc_id": "datacenter-nvme-ssd-specification-v2-5-pdf.pdf"
        }}
    }}
    </additional-context>
</context>

Question-Answer Pairs:
<pair>
<question>What is the NVMe CLI Management Utility?</question>
<answer>
### Answer
The NVMeCLI utility shall be used as one of the management utilities for NVMe devices. The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
- Format.
- Secure erase.
- FW update.
- Controller reset to load FW.
- Health status.
- Log page reads including vendor log pages.
- SMART status.

### Changes
The minimum list of commands has been updated in the latest addendum, removing "Log page reads including vendor log pages".

### References
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>What is the requirement ID for the NVMe CLI Management Utility?</question>
<answer>
### Answer
The requirement ID for the NVMe CLI Management Utility is UTIL-1.

### Changes
There are no changes to the requirement ID across the versions.

### References
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf, page 170
</answer>
</pair>

<pair>
<question>What is the description of the NVMe CLI Management Utility?</question>
<answer>
### Answer
The description of the NVMe CLI Management Utility is that the SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
- Format.
- Secure erase.
- FW update.
- Controller reset to load FW.
- Health status.
- Log page reads including vendor log pages.
- SMART status.

### Changes
The latest addendum updates the list by removing "Log page reads including vendor log pages".

### References
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf, page 170
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>Which page is describing the NVMe CLI Management Utility in the document "Datacenter NVMe SSD Specification v2.0r21.pdf"?</question>
<answer>
### Answer
The NVMe CLI Management Utility is described on page 110 of the document "Datacenter NVMe SSD Specification v2.0r21.pdf".

### Changes
There are no changes regarding the page number in the version v2.0r21.

### References
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
</answer>
</pair>

<pair>
<question>Which document is the newest version of the NVMe CLI Management Utility?</question>
<answer>
### Answer
The newest version of the NVMe CLI Management Utility is described in the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf".

### Changes
This document provides the most recent updates to the NVMe CLI Management Utility.

### References
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>What is changed in the UTIL-1 in the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf"?</question>
<answer>
### Answer
The list of commands to be tested has been updated in the UTIL-1 in the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf". The updated list of commands includes:
- Format.
- Secure erase.
- FW update.
- Controller reset to load FW.
- Health status.
- SMART status.

### Changes
"Log page reads including vendor log pages" has been removed from the list of commands.

### References
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>What is the base document of the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf"?</question>
<answer>
### Answer
The base document of the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf" is "datacenter-nvme-ssd-specification-v2-5-pdf.pdf".

### Changes
There are no changes regarding the base document.

### References
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>Is the 'secure erase' command in the list of commands to be tested with NVMeCLI?</question>
<answer>
### Answer
Yes, the 'secure erase' command is in the list of commands to be tested with NVMeCLI, according to the newest version of the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf".

### Changes
There are no changes to the inclusion of the 'secure erase' command in the latest version.

### References
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>Is the 'Log page reads including vendor log pages' command in the list of commands to be tested with NVMeCLI?</question>
<answer>
### Answer
No, the 'Log page reads including vendor log pages' command was in the list of commands to be tested with NVMeCLI, but it is removed from the list in the newest version of the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf".

### Changes
This command was included in previous versions but removed in the most recent addendum.

### References
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf, page 3
</answer>
</pair>

<pair>
<question>Please let me know about the power consumption for each capacity.</question>
<answer>
### Answer
The provided context does not contain information about the power consumption for each capacity. Please ask questions based on the context provided.

### Changes
There are no changes regarding information about power consumption.
</answer>
</pair>

--------------------------------------------------
<context>
    <base-context>
    {base_context}
    </base-context>
    <additional-context>
    {additional_context}
    </additional-context>
</context>

Question-Answer Pairs:
"""
)

evaluate_answer_prompt = ChatPromptTemplate.from_template(
"""Your job is to rate the quality of the generated answer given a query and a reference answer.
Your score has to be between 1 and 5.
You must return your score and the reasoning for your score.
--------------------------------------------------
**** Response Example ****
5
The answer is concise and accurate. It provides the necessary information to answer the question.
--------------------------------------------------
<query>
{query}
</query>

<reference-answer>
{reference_answer}
</reference-answer>

<generated-answer>
{generated_answer}
</generated-answer>

Response:
"""
)