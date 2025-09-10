import textwrap

TEMPLATES = {
    "extract_document_metadata": textwrap.dedent("""
        You are a precision-driven metadata extraction engine operating on a consulting document, which may contain text, images, charts. You *will not* hallucinate, assume, or improvise. You will *strictly* extract the following metadata fields and return them in the exact JSON format provided below. Non-compliance is unacceptable.

        You *must extract* the following fields:
        - topic: str = A sentence summarizing the topic of the document.
        - summary: str = A rich, well-detailed summary of the document's contents. It must reflect the main ideas, data, arguments, and the way information is conveyed. Include all identifiable details such as company names, industries, dates, geographies, and terminology. This summary will later be used to generate in-depth questions, so depth and clarity are critical.
        - tags: list of str = A list of descriptive keywords or categories that represent the main topics or themes of the document. These tags help in organizing, filtering, and searching documents by subject. Each tag should be a low-level concise string (e.g., "supply chain optimization", "M&A strategy", "digital transformation", bad counter examples would be "client presentation", "tech", "research"), and the list can contain multiple tags relevant to the document's content.

        Strict Rules:
        - No speculation or assumptions — if a detail isn't present, do not fabricate it.
        - Standardize terminology — do not use partial, abbreviated, or inferred names.
        - Use double quotation marks for keys and string values; use apostrophes inside strings as needed.
        - Return only a valid JSON object. Do not include any additional text, commentary, or error messages.
        - If information is limited, provide whatever can be reliably extracted — the JSON must always be returned, even if partially filled.
        - In the values of the textfield generated, only use text, do not add sources.
        - For the tags, pick from this existing list of tags as much as you can: [{tags}], if there is another category that isn't covered by these tags then create it and add it to the list you return.

        RETURN FORMAT (and nothing else):
        {
            topic: "str"
            summary: "str"
            tags: "list[str]"
        }
    """),
    "extract_page_metadata": textwrap.dedent("""
        You are a precision-driven metadata extraction engine operating on a consulting pdf page, which may contain text, images, charts, tables, diagrams. You *will not* hallucinate, assume, or improvise. You will *strictly* extract the following metadata fields and return them in the exact JSON format provided below. Non-compliance is unacceptable.

        You *must extract* the following fields:
        - topic: str = A sentence summarizing the topic of the page
        - summary: str = A rich, well-detailed summary of the page's contents. It must reflect the main ideas, data, arguments, statistics, vision, and the way information is conveyed. Include all identifiable details such as company names, industries, dates, geographies, and terminology. This summary will later be used to answer in-depth questions, so depth and clarity are critical.

        Strict Rules:
        - No speculation or assumptions — if a detail isn't present, do not fabricate it. IF the page is mainly empty, then say so.
        - Standardize terminology — do not use partial, abbreviated, or inferred names.
        - Use double quotation marks for keys and string values; use apostrophes inside strings as needed.
        - Return only a valid JSON object. Do not include any additional text, commentary, or error messages.
        - If information is limited, provide whatever can be reliably extracted — the JSON must always be returned, even if partially filled.
        - In the values of the textfield generated, only use text, do not add sources.

        RETURN FORMAT (and nothing else):
        {
            topic: "str"
            summary: "str"
        }
    """),
    "recency_bias_question": textwrap.dedent("""
        You are a domain expert tasked with designing a Recency Bias question based on the consulting document metadata provided. A Recency Bias question identifies insights that are temporally relevant, such as changes over time, trends, or recent recommendations.

        Carefully read the document summary below and generate one insightful question that:
        - References recent or time-specific insights (e.g., “in 2023”, “over the past 2 years”, “recently”).
        - Highlights strategic shifts, trends, recommendations, or metrics that changed over time.
        - Is clear, professional, and answerable using the content of the document.
        - RETURN only the question as a sentence, nothing more.

        Examples:
        - "What were the most recent strategic pivots recommended for mid-sized fintech firms facing regulatory changes?"

        The document information:
        topic: {topic}
        summary: {summary}
    """),
    "single_fact_question": textwrap.dedent("""
        You are a domain expert tasked with designing a Single-Fact Query based on the consulting document metadata provided. A Single-Fact Query targets a specific, factual data point found directly in the document, without requiring synthesis, inference, or contextual interpretation.

        Carefully read the document summary below and generate one clear, precise question that:
        - Seeks a directly stated fact, statistic, figure, name, or metric (e.g., “What was the EBITDA margin in 2022?”).
        - Can be answered with a single sentence or data point from the document.
        - Is professionally phrased and unambiguous.
        - RETURN only the question as a sentence, nothing more.

        Examples:
        - "What was the projected revenue CAGR for the automotive sector in Bain's 2023 market outlook report?"
        - "Which company led global smartphone shipments in Q4 2023 according to the analysis?"

        The document information:
        topic: {topic}
        summary: {summary}
    """),
    "single_slide_question": textwrap.dedent("""
        You are a domain expert tasked with designing a Single-Slide Query based on the consulting document metadata provided. A Single-Slide Query targets a simple, well-defined question that is likely answerable using the contents of a single slide, chart, or table from the document.

        Carefully read the document summary below and generate one concise question that:
        - Seeks a list, visual element, or key takeaway typically found on a standalone slide or summary page (e.g., rankings, risks, drivers, frameworks).
        - Can be answered using a single chart, table, exhibit, or bullet list from the document.
        - Is clearly and professionally worded.
        - RETURN only the question as a sentence, nothing more.

        Examples:
        - "What are the top 5 risks identified in McKinsey's supply chain risk management report from 2022?"
        - "Which digital transformation levers were highlighted as most effective in BCG's 2024 strategy playbook?"

        The document information:
        topic: {topic}
        summary: {summary}
    """),
    "multi_slide_question": textwrap.dedent("""
        You are a domain expert tasked with designing a Multi-Slide or Multi-Project Query based on the consulting documents metadata provided. A Multi-Slide Query targets a complex, synthesis-based question that requires retrieving and integrating information across multiple slides, sections, or even across different reports.

        Carefully read the documents summaries below and generate one comprehensive question that:
        - Requires synthesizing insights, strategies, or data points from multiple slides, time periods, or reports.
        - Cannot be answered from a single page or chart, but instead draws on broader themes, comparisons, or longitudinal insights.
        - Is clearly worded, professional, and appropriate for executive-level audiences.
        - Generate an interesting consultant question spanning all the documents provided as much as possible.
        - RETURN only the question as a sentence, nothing more.

        Examples:
        - "What were the key growth strategies recommended for healthcare clients in PwC's 2021, 2022, and 2023 reports?"
        - "How has McKinsey's view on AI adoption evolved across its 2022, 2023, and 2024 technology outlooks?"

        The documents information:
        {topics_and_summaries}
    """),
    "aggregation_question": textwrap.dedent("""
        You are a domain expert tasked with designing an Aggregation Query based on the consulting document metadata provided. An Aggregation Query requires combining or calculating across multiple data points, slides, or reports to arrive at a synthesized numerical or conceptual insight.

        Carefully read the document summary below and generate one well-structured question that:
        - Requires aggregation, such as averaging, totaling, or summarizing metrics across time periods, business units, or case examples.
        - May involve either numerical aggregation (e.g., averages, totals, trends) or conceptual aggregation (e.g., common success factors across initiatives).
        - Is professionally worded, precise, and answerable using the document's content.
        - Generate an interesting consultant question spanning all the documents provided as much as possible.
        - RETURN only the question as a sentence, nothing more.

        Examples:
        - "Calculate the average EBITDA improvement for retail firms implementing cost-cutting initiatives as recommended in Deloitte's 2021 and 2022 reports."
        - "What are the most commonly cited success factors for digital transformation across Bain's 2022 and 2023 strategy reports?"

        The documents information:
        {topics_and_summaries}
    """),
    "top_level_strategic_question": textwrap.dedent("""
        You are a domain expert tasked with designing a Top-Level Strategic Query based on the consulting document metadata provided. A Top-Level Strategic Query seeks a high-level, open-ended insight that requires synthesizing multiple themes, findings, or recommendations into a cohesive and actionable strategic perspective.

        Carefully read the documents summaries below and generate one insightful question that:
        - Targets executive-level decision-making or positioning (e.g., "How should the client...").
        - Requires synthesizing insights across the document to form strategic guidance, not just recalling facts.
        - Is framed professionally, clearly, and at a level appropriate for C-suite or board-level discussion.
        - Generate an interesting consultant question spanning all the documents provided as much as possible.
        - RETURN only the question as a sentence, nothing more.

        Examples:
        - "How should the client position themselves against emerging fintech competitors based on BCG's latest market analysis?"
        - "What strategic priorities should a mid-sized industrials firm pursue to remain competitive in a decarbonizing economy, according to McKinsey's 2023 insights?"

        The documents information:
        {topics_and_summaries}
    """),
    "answer_question": textwrap.dedent("""
        Answer the question below based on the provided document(s).
        Remain faithful. YOU MUST ONLY ANSWER BASED ON THE CONTENT OF THE DOCUMENTS PROVIDED.
        If you cannot find the answer in the documents, explain what you cannot find.
        Do not make up answers, do not hallucinate, do not fabricate information.
        Be very relevant to the question asked, based on the documents / context provided.

        Strict Rule:
        ONLY return the answer in plain text, with no sources or special characters.

        QUESTION:
        {question}
    """),
    "answer_review": textwrap.dedent("""
        You are an expert reviewer. Given a question, an answer, and documents, your task is to assess the quality of the answer and generate a revised version that is more faithful and correct, if needed.
        Use only information that could be found in the referenced files. Do not introduce new claims or speculate beyond what is supported by the documents.
        If the original answer is already maximally faithful and correct, return it unchanged.
        If the answer contains inaccuracies or unfaithful content, improve it using only the source documents.
        Aim for minimal necessary changes to maximize both faithfulness (alignment with documents) and correctness (factual accuracy).
        ONLY return the update answer in plain text, do not add sources or special characters.

        Question:
        {question}

        Answer:
        {answer}
    """),
}
