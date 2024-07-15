from langchain.prompts import ChatPromptTemplate

generation_prompt = ChatPromptTemplate.from_template(
"""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question 
considering the history of the conversation.

There may exist multiple documents containing the answer,
but they are different versions of the same document.
If you find multiple candidate answers,
you should compare them in detail and select the most recent one.

If you don't know the answer, just say that you don't know. 

You can answer in descriptive form or paraphrased form if you want,
and keep the answer concise.

You should answer in at most 4 sentences.

Feel free to use markdown to format your answer.


<chat-history>
{history}
</chat-history>


<context> 
{context} 
</context>


<question>
{query} 
</question>


Answer:"""
)

generation_prompt_v2 = ChatPromptTemplate.from_template(
"""
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question considering the chat history.

I have provided you with the base context and additional context.
Base context is the main context that you should consider first.
It is the most base information that contains the answer.
Additional context contains information about changes or updates to the base context, 
which are made according to the version updates of the documents or by the request of the specific clients.

You should answer step by step.
You should first find the answer from the base context,
and then consider the additional context to see if there are any updates or changes to the answer.
If there are several changes, you should consider the most recent additional context to provide the most up-to-date answer,
also comparing the changes to the base context.

If you don't know the answer, just say that you don't know.

You can answer in descriptive form or paraphrased form if you want, and keep the answer concise.

You should answer in at most 4 sentences.
Feel free to use markdown to format your answer.

--------------------------------------------------
**** Example 1 ****
<chat-history>
Human: Can you describe the features of UTIL-1?
</chat-history>

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

<question>
Can you describe the features of UTIL-1?
</question>

Answer:
### Answer
I can find the requirements for UTIL-1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and "datacenter-nvme-ssd-specification-v2-5-pdf.pdf". 
UTIL-1 describes that the ssd supplier must test their SSDs with the following utility and ensure compatibility. The followings are the minimum list of commands that should be tested with NVMeCLI.

- Format
- Secure erase
- FW update
- Controller reset to load FW
- Health status
- SMART status.
If you want to know more details, you can refer to the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf" page 170, which is the latest version of the document.

### Changes
The requirements for UTIL-1 is consistent in the documents between version 2.0r21 and 2.5.
The only difference is the section number, which is 16.1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and 18.1 in the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf".

However, the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf" provides an updated list of commands to be tested with NVMeCLI.
While the list of commands in the base document(datacenter-nvme-ssd-specification-v2-5-pdf.pdf) contains the command **Log page reads including vendor log pages**, it is removed in the updated list of commands in the addendum document.

### References
#### Base Documents
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf, page 170

#### Additional Documents
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20, page 3

--------------------------------------------------

<chat-history>
{history}
</chat-history>

<context>
    <base-context>
    {base_context}
    </base-context>
    <additional-context>
    {additional_context}
    </additional-context>
</context>

<question>
{query}
</question>

Answer:
"""
)

revision_prompt = ChatPromptTemplate.from_template(
"""You are an assistant for question-revision tasks.
Using given chat history,
rephrase the following question to be a standalone question.
The standalone question must have main words of the original question.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Revised question:
"""
)

hyde_prompt = ChatPromptTemplate.from_template(
"""You are an assistant for question-answering tasks.
Please write a passage to answer the question, considering the given chat history.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Answer:
"""
)