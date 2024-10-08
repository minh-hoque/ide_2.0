LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT = """
You are a highly skilled expert in evaluating language model responses with a focus on precision, detail, and contextual accuracy. Your task is to compare two responses: an old response, which is known to be high-quality, and a new response, which is generated by the model.

Your evaluation must determine whether the new response is equivalent to, or better than, the old response in the following areas:
1.	Quality: The clarity, coherence, and fluency of the response.
2.	Granularity: The level of detail and depth provided in the response.
3.	Relevance: The alignment of the response with the context and requirements of the prompt.

Instructions:
- If the new response matches or exceeds the old response across all three criteria, it should be deemed equivalent.
- If there are any significant deficiencies in the new response concerning quality, granularity, or relevance, it should be considered non-equivalent.
- You should provide a detailed rationale for your decision, explaining how the new response compares to the old one. This rationale is required.
- You must provide a result of either "ACCEPT" or "REJECT". This result is required.
-----------
Example:
- Old response: To apply for a loan, you need to visit our website and fill out the online application form. You will need to provide your income details, employment history, social security number, credit history, and additional documents, such as proof of address and identification, depending on the type of loan. Alternatively, you can apply via phone or by visiting one of our branches.
- New response: To apply for a loan, you can visit our website and complete the online application form. You’ll be required to submit details about your income, employment history, social security number, and credit history. Depending on the loan type, you may also need to provide additional documents like proof of address and identification. Alternatively, you have the option to apply over the phone or by visiting one of our branches.

Output:
{{
  "rationale": "The new response maintains the same level of quality, granularity, and relevance as the old response. It is clear, well-structured, and provides all the necessary information in a coherent manner. The slight variations in wording do not affect the overall equivalence.",
  "result": "ACCEPT"
}}
-----------
Output Format:
- The output should be in strict JSON format, containing two fields:
  1. "rationale": A clear and concise explanation of why the new response is "ACCEPT" or "REJECT". This should be a string.
  2. "result": "ACCEPT" if the new response is equivalent or better, or "REJECT" if it is not. This should be a single word between "ACCEPT" and "REJECT" with no additional formatting.
-----------
- Old response: {old_response}
- New response: {new_response}

Output:
{{
  "rationale": "",
  "result": ""
}}
"""

LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V2 = """
You are an highly skilled expert in evaluating language model responses. Your task is to compare two responses: an old response (known to be high-quality) and a new response generated by the model.

Determine if the new response is equivalent or better than the old response by checking if it respects the following Evaluation Criteria:

### Evaluation Criteria:
1. **Key Point Coverage**:
   - Ensure that the new response includes all essential key points from the old response.
   - Rephrasing or restructuring is acceptable, but no key points should be missing.
   - Additional information is acceptable only if it is accurate, relevant to the question, and enhances the response.

2. **Quality**:
   - Evaluate the clarity, coherence, and fluency of the new response compared to the old one.
   - The new response must have equal or improved clarity, coherence, and fluency as the old response.

3. **Detail and Depth**:
   - Assess the level of detail and depth provided in the new response.
   - The new response must provide equal or greater level of detail and depth compared to the old response


### Instructions:
- Analyze the old response and identify the key points required to answer the question.
- Check if the new response contains all key points present in the old response. Rephrasing or additional information is acceptable as long as it doesn’t omit, contradict, or dilute the key points.
- Ensure the new response matches or exceeds the old response in terms of clarity, coherence, and fluency.
- Check if the new response is of the same or more granular than the old response.
- Provide a brief rationale for your decision, explaining how the new response compares to the old one. This rationale is required.
- You must provide a result of either "ACCEPT" or "REJECT". This result is required.
-----------
### Examples:
#### Example 1:
Question: How do you bake a cake?
Old response: To bake a basic cake, you need flour, sugar, eggs, butter, and baking powder. Mix the dry ingredients, then add the wet ingredients. Pour the batter into a greased pan and bake at 350°F for 30 minutes.
New response: A simple cake recipe requires flour, sugar, eggs, butter, and baking powder. Combine dry ingredients first, then mix in wet ingredients. Bake the batter in a prepared pan at 350°F for about half an hour.

Output:
{{
  "rationale": "The new response includes all key points from the old response, such as the ingredients (flour, sugar, eggs, butter, and baking powder), the mixing method (dry first, then wet), the baking temperature (350°F), and the time (30 minutes). While the wording is rephrased, the meaning and instructions remain consistent. The additional detail specifying 'about half an hour' is acceptable as it aligns with the original time. The new response also maintains the same clarity and granularity as the old one, making it equally coherent and fluent.",
  "result": "ACCEPT"
}}

#### Example 2:
Question: What is the payment date schedule in this legal document?
Old response: The payment schedule outlined in the document specifies that payments are due on the 15th of every month, starting from March 2024. The payments will continue on the 15th of each month for a period of 24 months, concluding in February 2026. Any delays in payment will incur a penalty of 2% of the outstanding balance for each week of delay.
New response: The legal document states that payments are due on the 15th of every month starting in March 2024. The schedule continues for 24 months, ending in February 2026. Additionally, a 2% penalty applies to late payments for each week of delay.

Output:
{{
  "rationale": "The new response contains all key points from the old response, including the payment due date (15th of each month), the starting month (March 2024), the duration of payments (24 months), and the penalty clause (2% for each week of delay). While the new response is more concise, it conveys the same critical information without omitting any important details. The clarity and structure of the new response are on par with the old response, and it offers the same level of granularity in terms of both schedule and penalties.",
  "result": "ACCEPT"
}}

#### Example 3:
Question: What are the health benefits of regular exercise?
Old response: Regular exercise improves cardiovascular health, aids in weight management, enhances mental health by reducing stress and anxiety, and strengthens muscles and bones. It is recommended to engage in at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity each week.
New response: Exercise helps with stress reduction and can improve physical health by strengthening the heart. It’s good to stay active.

Output:
{{
  "rationale": "The new response significantly lacks the detail and depth present in the old response. Key points such as the specific health benefits of exercise like weight management, mental health improvements, and strengthening of muscles and bones are omitted. Additionally, the old response's recommendations for weekly activity levels are completely missing. The new response's brevity and lack of comprehensive information significantly reduce its value compared to the old response, failing to meet the criteria of key point coverage and granularity.",
  "result": "REJECT"
}}
-----------
### Output Format:
- Provide output in strict JSON format with two fields:
  1. "rationale": A concise explanation of your decision (string).
  2. "result": Either "ACCEPT" or "REJECT" (single word, no formatting).
-----------
### Input:
Question: {question}
Old response: {old_response}
New response: {new_response}

Output:
{{
  "rationale": "",
  "result": ""
}}
"""

LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V3 = """
You are an highly skilled expert in evaluating language model responses. Your task is to compare two responses: an old response (known to be high-quality) and a new response generated by the model.

Determine if the new response is equivalent or better than the old response by checking if it respects the following Evaluation Criteria:

### Evaluation Criteria:
1. **Key Point Coverage**:
   - Ensure that the new response includes all essential key points from the old response that is important to answer the question.
   - Rephrasing or restructuring is acceptable, but no key points should be missing.
   - Additional key points in the new response that is not in the old response are not acceptable. This is because it might be hallucination or inaccurate.

2. **Quality**:
   - Evaluate the clarity, coherence, and fluency of the new response compared to the old one.
   - The new response must have equal or improved clarity, coherence, and fluency as the old response.


### Instructions:
- Analyze the old response and identify the key points required to answer the question.
- Check if the new response contains all key points present in the old response. Rephrasing or restructuring is acceptable as long as it doesn’t omit, contradict, or dilute the key points.
- Check if the new response has any key points that are not in the old response. If it does, it should be rejected. This is because it might be hallucination or inaccurate.
- Ensure the new response matches or exceeds the old response in terms of clarity, coherence, and fluency.
- Provide a brief rationale for your decision, explaining how the new response compares to the old one. This rationale is required.
- You must provide a result of either "ACCEPT" or "REJECT". This result is required.
-----------
### Examples:
#### Example 1:
Question: How do you bake a cake?
Old response: To bake a basic cake, you need flour, sugar, eggs, butter, and baking powder. Mix the dry ingredients, then add the wet ingredients. Pour the batter into a greased pan and bake at 350°F for 30 minutes.
New response: A simple cake recipe requires flour, sugar, eggs, butter, and baking powder. Combine dry ingredients first, then mix in wet ingredients. Bake the batter in a prepared pan at 350°F for about half an hour.

Output:
{{
  "rationale": "The new response includes all key points from the old response, such as the ingredients (flour, sugar, eggs, butter, and baking powder), the mixing method (dry first, then wet), the baking temperature (350°F), and the time (30 minutes). While the wording is rephrased, the meaning and instructions remain consistent. The additional detail specifying 'about half an hour' is acceptable as it aligns with the original time. The new response also maintains the same clarity and granularity as the old one, making it equally coherent and fluent.",
  "result": "ACCEPT"
}}

#### Example 2:
Question: What is the payment date schedule in this legal document?
Old response: The payment schedule outlined in the document specifies that payments are due on the 15th of every month, starting from March 2024. The payments will continue on the 15th of each month for a period of 24 months, concluding in February 2026. Any delays in payment will incur a penalty of 2% of the outstanding balance for each week of delay.
New response: The legal document states that payments are due on the 15th of every month starting in March 2024. The schedule continues for 24 months, ending in February 2026. Additionally, a 2% penalty applies to late payments for each week of delay.

Output:
{{
  "rationale": "The new response contains all key points from the old response, including the payment due date (15th of each month), the starting month (March 2024), the duration of payments (24 months), and the penalty clause (2% for each week of delay). While the new response is more concise, it conveys the same critical information without omitting any important details. The clarity and structure of the new response are on par with the old response, and it offers the same level of granularity in terms of both schedule and penalties.",
  "result": "ACCEPT"
}}
-----------
### Output Format:
- Provide output in strict JSON format with two fields:
  1. "rationale": A concise explanation of your decision (string).
  2. "result": Either "ACCEPT" or "REJECT" (single word, no formatting).
-----------
### Input:
Question: {question}
Old response: {old_response}
New response: {new_response}

Output:
{{
  "rationale": "",
  "result": ""
}}
"""

LLM_AS_A_JUDGE_EQUIVALENCE_PROMPT_V4 = """
Your Task: Evaluate two model responses: a reference response validated by a Subject Matter Expert (SME) that is known to be of high quality and a new response generated by the model.
Your Goal: Provide a data scientist a quick evaluation of the new response's quality based on the SME evaluation of the old response. Determine if the new response is equivalent or better than the old reference response by checking if it respects the following Evaluation Criteria:

### Evaluation Criteria:
1. **Key Point Coverage**:
   - Identify Key Points: Review the Reference response to identify the mandatory critical information that effectively answers the question.
   - Verify Key Point Presence: Ensure the new response contains all identified key points. Acceptable modifications include rephrasing or restructuring as long as these changes do not affect the factual content.
   - Reject Additional Content: Any new content not present in the Reference response should lead to rejection, except minor polite additions like “Thank you” or elaborations that clarify but do not introduce new unverifiable information.

2. **Irrelevance of Tone and Style:**:
   - Ignore the tone and style; focus solely on the factual and informative content alignment between the reference and new response.

### Instructions:
- Identify Key Points: Review the Reference response to identify the mandatory key points that effectively answer the question.
- Verify Key Point Presence: Ensure the new response contains all identified key points. Acceptable modifications include rephrasing or restructuring as long as these changes do not affect the factual content.
- Reject Additional Content: Any new content not present in the Reference response should lead to rejection, except minor polite additions like “Thank you” or elaborations that clarify but do not introduce new facts.
- Provide a brief rationale for your decision, explaining how the new response compares to the old one. This rationale is required.
- You must provide a result of either "ACCEPT" or "REJECT". This result is required.
-----------
### Examples:
#### Example 1:
Question: How do you bake a cake?
Reference response: To bake a basic cake, you need flour, sugar, eggs, butter, and baking powder. Mix the dry ingredients, then add the wet ingredients. Pour the batter into a greased pan and bake at 350°F for 30 minutes.
New response: A simple cake recipe requires flour, sugar, eggs, butter, and baking powder. Combine dry ingredients first, then mix in wet ingredients. Bake the batter in a prepared pan at 350°F for about half an hour.

Output:
{{
  "rationale": "All key points about ingredients and steps are preserved with minor rephrasing for clarity. The slight change in wording for the baking time ('about half an hour') is acceptable as it does not alter the factual content.",
  "result": "ACCEPT"
}}

#### Example 2:
Question: What is the payment date schedule in this legal document?
Reference response: Payments are due on the 15th of every month, starting from March 2024, for 24 months, concluding in February 2026. A 2% penalty applies for each week of delay.
New response: Payments start in March 2024, due monthly until February 2026. Late payments attract a weekly penalty of 2%.

Output:
{{
  "rationale": "The new response omits the specific due date (15th of each month), failing to fully capture the essential timing details from the old response.",
  "result": "REJECT"
}}

#### Example 3:
Question: What are the symptoms of the common cold?
Reference response: Common cold symptoms include runny or stuffy nose, sore throat, cough, mild headache, and sometimes a low-grade fever.
New response: The common cold typically causes a runny or congested nose, sore throat, coughing, and mild headache. In some cases, patients may experience a low-grade fever. It's important to note that these symptoms usually resolve within 7-10 days.

Output:
{{
  "rationale": "While the new response accurately covers all key symptoms mentioned in the reference response, it introduces additional information about symptom duration (7-10 days) that is not present in the reference response. This new information, although potentially accurate, cannot be verified against the reference and could be a hallucination.",
  "result": "REJECT"
}}

#### Example 4:
Question: What is the process of photosynthesis?
Reference response: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It involves the green pigment chlorophyll and generates oxygen as a byproduct.
New response: In photosynthesis, plants convert carbon dioxide and water into food using sunlight. This process primarily occurs in the leaves and involves chlorophyll. Oxygen is released as a byproduct.

Output:
{{
  "rationale": "The new response covers all essential points from the reference response, such as the involvement of sunlight, carbon dioxide, water, chlorophyll, and the release of oxygen. The simplification and restructuring of the information do not omit or misrepresent the critical facts. The new response maintains the clarity and essential content of the reference.",
  "result": "ACCEPT"
}}

#### Example 5:
Question: What are the main causes of global warming?
Reference response: The main causes of global warming are the increase in greenhouse gases from human activities, such as deforestation, burning fossil fuels, and industrial processes.
New response: Global warming is primarily caused by natural phenomena such as volcanic eruptions and variations in solar radiation. Human activities like fossil fuel combustion exacerbate these natural effects.

Output:
{{
  "rationale": "The new response shifts the primary causes of global warming to natural phenomena, which contradicts the established key points in the reference response that attribute the main causes to human activities. This substantial deviation misrepresents the scientifically accepted understanding and could mislead based on the reference criteria.",
  "result": "REJECT"
}}
-----------
### Output Format:
- Provide output in strict JSON format with two fields:
  1. "rationale": A concise explanation of your decision (string).
  2. "result": Either "ACCEPT" or "REJECT" (single word, no formatting).
-----------
### Input:
Question: {question}
Reference response: {old_response}
New response: {new_response}

Output:
{{
  "rationale": "",
  "result": ""
}}
"""


LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT = """
You are a highly skilled expert in evaluating language model responses, particularly in assessing the incorporation of subject matter expert (SME) feedback. Your task is to compare an old response, which includes SME feedback, with a new response generated by the model.

Your evaluation must determine whether the new response has effectively incorporated the SME feedback in the following areas:
1.	Accuracy: The extent to which the new response reflects the corrections or enhancements suggested by the SME.
2.	Completeness: Whether all aspects of the SME feedback have been addressed in the new response.
3.	Relevance: The degree to which the SME feedback has been integrated appropriately within the context of the original response.

Instructions:
- If the new response fully incorporates the SME feedback across all criteria, it should be accepted.
- If any significant aspects of the SME feedback are missing or inadequately addressed, the new response should be "REJECT".
- You should provide a detailed rationale for your decision, explaining how the new response applies or rejects the SME feedback. This rationale is required.
- You must provide a result of either "ACCEPT" or "REJECT". This result is required.
-----------
Example:
- Old response: You can reset your password by going to the settings page and clicking on "Forgot Password?"
- SME feedback: This is not a complete answer. It should mention alternatives to resetting the password such as calling customer support.
- New response: You can reset your password by going to the settings page and clicking on "Forgot Password?". You can also call customer support to reset your password.

Output:
{{
  "rationale": "The new response effectively incorporates the SME feedback by mentioning the alternative method of contacting customer support for password reset. The response is now complete and provides users with multiple options, reflecting the SME's suggestions.",
  "result": "ACCEPT"
}}
-----------
Output Format:
- The output should be in strict JSON format, containing two fields:
  1. "Rationale": A clear and concise explanation of why the new response is "ACCEPT" or "REJECT". This should be a string.
  2. "Result": "ACCEPT" if the SME feedback is fully incorporated, or "REJECT" if it is not. This should be a single word between "ACCEPT" and "REJECT" with no additional formatting.
-----------
- Old response: {old_response}
- SME feedback: {sme_feedback}
- New response: {new_response}

Output:
{{
  "rationale": "",
  "result": ""
}}
"""


LLM_AS_A_JUDGE_SME_FEEDBACK_PROMPT_V2 = """
You are a highly skilled expert in evaluating language model responses, particularly in assessing the incorporation of subject matter expert (SME) feedback. Your task is to compare an old response, which has received SME feedback, with a new response generated by the model.

Your evaluation must determine whether the new response has effectively incorporated the SME feedback based on the following criteria:
1.	Accuracy: The extent to which the new response reflects the corrections or enhancements suggested by the SME.
2.	Completeness: Whether all aspects of the SME feedback have been addressed in the new response.
3.	Relevance: The degree to which the SME feedback has been integrated appropriately within the context of the original response.

### Instructions:
- Accept the new response if it fully incorporates the SME feedback across all criteria.
- Reject the new response if any significant aspect of the SME feedback is missing or inadequately addressed.
- Provide a clear rationale for your decision, explaining how well the new response applies the SME feedback. This rationale is required.
- You must provide a result of either "ACCEPT" or "REJECT." This result is required.
-----------
### Example:
- Question: How do I reset my password?
- Old response: You can reset your password by going to the settings page and clicking on "Forgot Password?"
- SME feedback: This is not a complete answer. It should mention alternatives to resetting the password such as calling customer support.
- New response: You can reset your password by going to the settings page and clicking on "Forgot Password?". You can also call customer support to reset your password.

Output:
{{
  "rationale": "The new response effectively incorporates the SME feedback by mentioning the alternative method of contacting customer support for password reset. The response is now complete and provides users with multiple options, reflecting the SME's suggestions.",
  "result": "ACCEPT"
}}
-----------
### Output Format:
- The output should be in strict JSON format, containing two fields:
  1. "Rationale": A clear and concise explanation of why the new response is "ACCEPT" or "REJECT". This should be a string.
  2. "Result": "ACCEPT" if the SME feedback is fully incorporated, or "REJECT" if it is not. This should be a single word between "ACCEPT" and "REJECT" with no additional formatting.
-----------
### Input:
- Question: {question}
- Old response: {old_response}
- SME feedback: {sme_feedback}
- New response: {new_response}

Output:
{{
  "rationale": "",
  "result": ""
}}
"""
