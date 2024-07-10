"""
1) Connection
2) Save output in excel
3) Prompt-engineering
 
"""
 
import os
from llm_commons.langchain.proxy import ChatOpenAI, OpenAI, HuggingFaceTextGenInference
from llm_commons.btp_llm.identity import BTPProxyClient
import json
import pandas as pd

from llm_commons.proxy.base import set_proxy_version
from llm_commons.langchain.proxy import init_llm, init_embedding_model
set_proxy_version('btp')
 
btp_proxy_client = BTPProxyClient()
 
input_prompt = """
You are a world class summarizer. I am a customer of this service named "MyS4HANACLOUD_01". I wanted to know the status of my service till this timeframe and the monitoring data has been provided. Provide a qualitative summary of the below data of my service in a neat understandable format in bulleted points. I also wanted to know the overall status of my service. Provide me the overall status in simple terms.
Health Monitoring Data:
[
    {
        "Rating": "Information",
        "Text": "Application jobs delay",
        "Value": "0 s"
    },
    {
        "Rating": "Ok",
        "Text": "Number of ABAP dumps during the last 5 minutes",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "29.51 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "29.11 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "39.39 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "284.76 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Total Application Logs Count by Object",
        "Value": "237 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "140.56 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "38493.58 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "2 "
    },
    {
        "Rating": "Ok",
        "Text": "Total Application Logs Count by Object",
        "Value": "87 "
    },
    {
        "Rating": "Information",
        "Text": "Application jobs executions count",
        "Value": "1 "
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "52.3 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Total Application Logs Count by Object",
        "Value": "1613 "
    },
    {
        "Rating": "Critical",
        "Text": "Number of ABAP dumps during the last 5 minutes",
        "Value": "5 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "448.33 MB"
    },
    {
        "Rating": "Information",
        "Text": "Number of active sessions in the system",
        "Value": "10 "
    },
    {
        "Rating": "Critical",
        "Text": "Number of ABAP dumps during the last 5 minutes",
        "Value": "2 "
    },
    {
        "Rating": "Critical",
        "Text": "Number of ABAP dumps during the last 5 minutes",
        "Value": "11 "
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "21.98 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "0 "
    },
    {
        "Rating": "Critical",
        "Text": "Number of ABAP dumps during the last 5 minutes",
        "Value": "1 "
    },
    {
        "Rating": "Ok",
        "Text": "Work Process Utilization",
        "Value": "0.83 %"
    },
    {
        "Rating": "Ok",
        "Text": "Total Application Logs Count by Object",
        "Value": "398 "
    },
    {
        "Rating": "Ok",
        "Text": "Application jobs executions count",
        "Value": "0 "
    },
    {
        "Rating": "Critical",
        "Text": "Client Certificate Expiry",
        "Value": "-141 days"
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "1 "
    },
    {
        "Rating": "Ok",
        "Text": "Application jobs executions count",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "50.38 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Top Table Size by App Hierarchy",
        "Value": "27.12 MB"
    },
    {
        "Rating": "Ok",
        "Text": "Application Logs Count Having Errors by Object of Last 5 Minutes",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Total Application Logs Count by Object",
        "Value": "146 "
    },
    {
        "Rating": "Ok",
        "Text": "HANA Memory Used",
        "Value": "115.37 GB"
    },
    {
        "Rating": "Information",
        "Text": "Application jobs executions count",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Locked Users BUS",
        "Value": "0 "
    },
    {
        "Rating": "Ok",
        "Text": "Total Application Logs Count by Object",
        "Value": "250 "
    }
]

Integration Monitoring Data:
[
    {
        ""Status"": ""E"",
        ""Message"": ""Failed to lock control record when sending Work Package PROJ_1601885207977 to ONEmds EPPM_0581 entity Project Controlling Object."",
        ""User"": ""CC0000000010""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980004575"",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""No corresponding supplier could be found for business partner 98456148957"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""OAuth / destination error when sending Work Package PROJ_1601885271004 to ONEmds EPPM_0581 with entity Project Controlling Object."",
        ""User"": ""CC0000000010""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001443"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""Failed to lock control record when sending Work Package PROJ_1601885207977 to ONEmds EPPM_0581 entity Project Controlling Object."",
        ""User"": ""CC0000000010""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980002206"",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""Failed to lock control record when sending Work Package PROJ_1601885298665_2 to ONEmds EPPM_0581 entity Project Controlling Object."",
        ""User"": ""CC0000000010""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001443"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""Objects with errors were stored. Repeat replication for those objects."",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""Failed to lock control record when sending Work Package PROJ_1601885207977 to ONEmds EPPM_0581 entity Project Controlling Object."",
        ""User"": ""CC0000000010""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""OAuth / destination error when sending Work Package PROJ_1601885271004 to ONEmds EPPM_0581 with entity Project Controlling Object."",
        ""User"": ""CC0000000010""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980004575"",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980002206"",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""Objects with errors were stored. Repeat replication for those objects."",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""No corresponding supplier could be found for business partner 98456148957"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""No corresponding supplier could be found for business partner 98456148957"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""Objects with errors were stored. Repeat replication for those objects."",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 36100003"",
        ""User"": ""CB9980003720""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001444"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001445"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001443"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 36100003"",
        ""User"": ""CB9980003720""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001443"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""No corresponding supplier could be found for business partner 82754746598"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001444"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980002206"",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001445"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""Objects with errors were stored. Repeat replication for those objects."",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001445"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""No corresponding supplier could be found for business partner 98456148957"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001445"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""Objects with errors were stored. Repeat replication for those objects."",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001443"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980004575"",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""No corresponding supplier could be found for business partner 82754746598"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""W"",
        ""Message"": ""Objects with errors were stored. Repeat replication for those objects."",
        ""User"": ""SAP_SYSTEM""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001445"",
        ""User"": ""CC0000000037""
    },
    {
        ""Status"": ""E"",
        ""Message"": ""No corresponding supplier could be found for business partner 9980001445"",
        ""User"": ""CC0000000037""
    },
]

Job Monitoring data:
[
    {
        ""Name"": ""Leave Request Approval (WS8646935)"",
        ""Execution Status"": ""Success"",
        ""Application Status"": ""No Data"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 0
    },
    {
        ""Name"": ""Material Requirements Planning (MRP)"",
        ""Execution Status"": ""Success"",
        ""Application Status"": ""Success"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 0
    },
    {
        ""Name"": ""FI-CA Payment Run"",
        ""Execution Status"": ""Success"",
        ""Application Status"": ""Success"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 0
    },
    {
        ""Name"": ""Invoicing (Mass Creation)"",
        ""Execution Status"": ""Success"",
        ""Application Status"": ""Success"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 0
    },
    {
        ""Name"": ""Change Request Processing (WS575828209)"",
        ""Execution Status"": ""Success"",
        ""Application Status"": ""Success"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 0
    },
    {
        ""Name"": ""Master Data Update"",
        ""Execution Status"": ""Error"",
        ""Application Status"": ""Success"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 1
    },
    {
        ""Name"": ""Actual Overhead Calculation: Service Documents – Parallel Ledgers"",
        ""Execution Status"": ""Error"",
        ""Application Status"": ""Error"",
        ""Start Delay"": ""Success"",
        ""Run Time"": ""Success"",
        ""Open Alerts"": 2
    }
]
"""
 
# btp_llm = ChatOpenAI(proxy_client=btp_proxy_client, deployment_id="gpt-4-32k", temperature=0, max_tokens=800)
# input_prompt = """
# "Identify any metric or exception that might have correlation between the different monitoring use-cases in the below data provided.
 
# Health monitoring summary:
# The service has a variety of metrics, with most of them rated as ""Ok"". Here's a summary of the key metrics:
# 1. Application jobs delay: The delay in application jobs is 0 seconds, which is excellent.
# 2. Number of ABAP dumps during the last 5 minutes: There have been several instances of ABAP dumps, with a maximum of 11 in the last 5 minutes. This is a critical issue and needs immediate attention.
# 3. Top Table Size by App Hierarchy: The table sizes vary, with the largest being 38493.58 MB.
# 4. Application Logs Count Having Errors by Object of Last 5 Minutes: There are a few instances of application logs having errors, with a maximum of 2 in the last 5 minutes.
# 5. Total Application Logs Count by Object: The total application logs count by object varies, with the highest being 2825011.
# 6. Application jobs executions count: The count of application job executions is mostly 0 or 1.
# 7. Number of active sessions in the system: There are 10 active sessions in the system.
# 8. Client Certificate Expiry: The client certificate has expired and is overdue by 151 days. This is a critical issue and needs immediate attention.
# 9. Work Process Utilization: The work process utilization is low, with a maximum of 20.37%.
# 10. HANA Memory Used: The HANA memory used is 115.37 GB.
# 11. Number of active users in the system: There are 4 active users in the system.
# Overall, the service seems to be functioning well in most areas, but there are critical issues with ABAP dumps and client certificate expiry that need to be addressed immediately.
# Given the critical issues, the overall health rating of the service would be around 60 out of 100. This rating could be improved by addressing the critical issues and monitoring the service for any new issues.
 
# Integration monitoring summary:
# During a day of service for ""MyS4HanaCloud_01"" with service id ""1"", several exceptions occurred:
# 1. There were multiple instances of a failure to lock control records when sending work packages to ONEmds EPPM_0581 entity Project Controlling Object. This error was encountered by user ""CC0000000010"".
# 2. There were numerous instances of the system not being able to find a corresponding supplier for various business partners. This error was encountered by users ""SAP_SYSTEM"", ""CC0000000037"", and ""CB9980003720"".
# 3. There were several OAuth / destination errors when sending work packages to ONEmds EPPM_0581 with entity Project Controlling Object. This error was encountered by user ""CC0000000010"".
# 4. There were multiple instances of objects with errors being stored, with the suggestion to repeat replication for those objects. This error was encountered by user ""SAP_SYSTEM"".
# The status of these exceptions varied between ""E"" (Error) and ""W"" (Warning).'"
 
# """



max_tokens = 800
temperature = 0
 
all_llms = {
    'text-davinci-003':  OpenAI(deployment_id='text-davinci-003', max_tokens = max_tokens, temperature = temperature),
    'gpt-35-turbo':  ChatOpenAI(deployment_id='gpt-35-turbo', max_tokens = max_tokens, temperature = temperature),
    'gpt-4':  ChatOpenAI(deployment_id='gpt-4', max_tokens = max_tokens, temperature = temperature),
    'gpt-4-32k':  ChatOpenAI(deployment_id='gpt-4-32k', max_tokens = max_tokens, temperature = temperature),
    # 'alephalpha':  AlephAlpha(deployment_id='alephalpha', maximum_tokens=max_tokens, temperature=temperature),
    # 'anthropic-claude-v1':  BedrockChat(deployment_id='anthropic-claude-v1', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-v2':  BedrockChat(deployment_id='anthropic-claude-v2', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-instant-v1':  BedrockChat(deployment_id='anthropic-claude-instant-v1', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-v1-100k':  BedrockChat(deployment_id='anthropic-claude-v1-100k', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-v2-100k':  BedrockChat(deployment_id='anthropic-claude-v2-100k', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-direct-claude-instant-1':  BedrockChat(deployment_id='anthropic-direct', model_kwargs={'model': 'claude-instant-1', "temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-direct-claude-2':  BedrockChat(deployment_id='anthropic-direct', model_kwargs={'model': 'claude-2', "temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'ai21-j2-grande-instruct':  Bedrock(deployment_id='ai21-j2-grande-instruct', model_kwargs={"temperature": temperature, "maxTokens": max_tokens}),
    # 'ai21-j2-jumbo-instruct':  Bedrock(deployment_id='ai21-j2-jumbo-instruct', model_kwargs={"temperature": temperature, "maxTokens": max_tokens}),
    # 'amazon-titan-tg1-large':  Bedrock(deployment_id='amazon-titan-tg1-large', model_kwargs={"temperature": temperature}),
    # 'gcp-text-bison-001':  GooglePalm(deployment_id='gcp-text-bison-001', temperature = temperature, max_output_tokens = max_tokens),
    # 'falcon-7b':  HuggingFaceTextGenInference(deployment_id='falcon-7b'),#, temperature = temperature, max_new_tokens = max_tokens), # issue https://github.tools.sap/AI-Playground-Projects/llm-commons/issues/106 - size limit
    'falcon-40b-instruct':  HuggingFaceTextGenInference(deployment_id='falcon-40b-instruct'),#, temperature = temperature, max_new_tokens = max_tokens),
    # 'llama2-13b-chat-hf':  HuggingFaceTextGenInference(deployment_id='llama2-13b-chat-hf')#, temperature = temperature, max_new_tokens = max_tokens), - size limit
}
 
df = pd.DataFrame(columns=['input_prompt'] + list(all_llms.keys()))
row_data = {'input_prompt': input_prompt}
output_str=""
 
for llm_key, llm_endpt in all_llms.items():
    try:
        print(llm_key)
        if (llm_key=='text-davinci-003' or llm_key=='falcon-7b' or llm_key=='falcon-40b-instruct' or llm_key=='llama2-13b-chat-hf'):
            output_str = str(llm_endpt.invoke(input_prompt))
        else:
            output_str = str(llm_endpt.invoke(input_prompt).content)
        # parsed_json = json.loads(output_str)
        # pretty_json = json.dumps(parsed_json, indent=2)
        row_data[llm_key] = output_str
        # # Print the readable JSON
        # print(pretty_json)
       
        print(output_str)
    except:
        output_str = "Prompt size too large"
        row_data[llm_key] = output_str
df = pd.concat([df,pd.DataFrame([row_data])], ignore_index=True)
print(df)
excel_loc = "/Users/I587795/Desktop/GenAI@CALM/Test_Summarization.xlsx"
try:
    # Try to read the existing Excel file
    df_existing = pd.read_excel(excel_loc, sheet_name="Sheet1")
       
    # Append the new data to the existing DataFrame
    df_combined = pd.concat([df_existing, df], ignore_index=True)
except FileNotFoundError:
    # If the file doesn't exist, create a new DataFrame with the new data
    df_combined = df
 
    # Write the combined DataFrame back to the Excel file
# df_combined.to_excel("C:/Users/I587795/OneDrive - SAP SE/CALM_GenAI_PromptResponses.xlsx", index=False, sheet_name="Sheet 1")
with pd.ExcelWriter(excel_loc, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    df_combined.to_excel(writer, index=False, sheet_name="Sheet1")
