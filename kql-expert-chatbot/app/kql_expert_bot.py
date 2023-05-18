from typing import List
from pbyc.engines import PbyCEngine
from pbyc.representations import PbyCRepresentation
from pbyc.tools.aoi_chat import AzureChatOpenAITool
from pbyc.tools.file import FileTool
from pbyc.tools import PbyCTool
from pbyc.types import Representation
from pbyc.types import ChangedRepresentation, Representation, Response, ChatMessage
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
import aiohttp
import tiktoken
import json

class KqlExpertBotTool(PbyCEngine):
    def _get_representations(self):
        return [SchemaRep(), ConfigRep()]

    def get_tokensize(self, text):
        enc = tiktoken.encoding_for_model('text-davinci-003')
        return len(enc.encode(text))
    
    async def _take_utterance(self, text:str, **kwargs):
        tool = CommandChooser(credentials=self._credentials)
        cmd = await tool.run(
            SCHEMA=self._project.representations["schema"].text,
            CONFIG=self._project.representations["config"].text,
            user_input=text
        )

        change_summary, config = self.process_output(cmd)
        if config != "":
            self._project.representations["config"].text = config
            configs = json.loads(config)

            schema_from_kusto = []

            if configs["KUSTO_CLUSTER_URL"] != "" and configs["KUSTO_DATABASE_NAME"] != "" and configs["TENANT_ID"] != "" and configs["SERVICE_CLIENT_ID"] != "" and configs["SERVICE_CLIENT_SECRET"] != "" :
                schema_from_kusto = self.get_schema_from_kusto(configs)

            if schema_from_kusto:
                schema = schema_from_kusto
                self._project.representations["schema"].text = str(schema)
        

        await self._progress(
            Response(
                type="output",
                message=change_summary,
                project=self._project
            ))
        return change_summary + "\n[Config]\n" + config
    
    def get_schema_from_kusto(self, configs:dict):

        cluster = configs["KUSTO_CLUSTER_URL"]  # Replace with your cluster URL
        database = configs["KUSTO_DATABASE_NAME"]  # Replace with your database name
        authority_id = configs["TENANT_ID"]  # Replace with your authority ID (e.g., organizations)
        client_id = configs["SERVICE_CLIENT_ID"]  # Replace with your service principal client ID
        client_secret = configs["SERVICE_CLIENT_SECRET"]  # Replace with your service principal client secret

        kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
            cluster, client_id, client_secret, authority_id
        )

        schemas = dict()

        try:
            client = KustoClient(kcsb)
            query = ".show tables"
            response = client.execute(database, query)
            tables = []
            
            # Process the response
            for row in response.primary_results[0]:
                tables.append(row["TableName"])

            print("#################Tables loaded##############")
            print(f"Tables: {tables}")

            for table in tables:
                schemaQuery = ".show table {0} cslschema".format(table)
                intermediateResponse = client.execute(database, schemaQuery)
                for row in intermediateResponse.primary_results[0]:
                    schemas[table] = row["Schema"]
                

            print(f"Schemas: {schemas}")
                
        except KustoServiceError as error:
            print(f"Kusto service error occurred: {error}")

        return schemas


    async def _get_output(self, text:str, **kwargs):

        chat_history = kwargs.get("chat_history", [])

        tool = OutputBot(credentials=self._credentials)
        output = await tool.run(
            SCHEMA=self._project.representations["schema"].text,
            CONFIG=self._project.representations["config"].text,
            chat_history=chat_history,
            user_input=text)
        
        response_index = -1
        for idx, line in enumerate(output):
            if '[Response]' in line or 'Response:' in line:
                response_index = idx
        
        if response_index == -1:
            response = output
        else:
            response = '\n'.join(output[response_index + 1:])
        #self._project.representations['variable_values'].text = variables

        await self._progress(Response(
                type="output",
                message=response,
                project=self._project
            ))

        return response
    
    async def _take_representation_edit(self, edit: ChangedRepresentation, **kwargs):

        if edit.name in self._project.representations:
            self._project.representations[edit.name].text = edit.text
        
        await self._progress(Response(
                type="output",
                message='Schema modified',
                project=self._project
            ))

    def process_output(self, output):
        output = output.split('\n')
        print(output)
        config_index_prev = -1
        config_index = -1
        for idx, line in enumerate(output):
            if 'summary' in line:
                change_summary = line.split(':')[1].strip()
            elif '[CONFIG]' in line or 'CONFIG:' in line:
                config_index_prev = idx
            elif 'KUSTO_CLUSTER_URL' in line:
                config_index = idx

        config = ""
        if config_index_prev != -1:
            config = '\n'.join(output[config_index_prev + 1:])
        if config_index != -1:
            config = '\n'.join(output[config_index:])
        return change_summary, config
    
class SchemaRep(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="schema",
            text="",
            type="md"
        )
    
class ConfigRep(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="config",
            text="{\"KUSTO_CLUSTER_URL\":\"\",\"KUSTO_DATABASE_NAME\":\"\",\"SERVICE_CLIENT_ID\":\"\",\"SERVICE_CLIENT_SECRET\":\"\",\"TENANT_ID\":\"\"}",
            type="json"
        )

class OutputBot(AzureChatOpenAITool):
    def _get_system_prompt(self):
        return """
You are a KQL expert that can write queries based on tables schemas present in schema [SCHEMA] and can display the current configuration present in config [CONFIG]. 

The schema is a set of key-value pairs, where the keys are table names (which you can think of as KQL tables), values are schema of that table. We want the schema section names to be distinct from each other. 

The config is a json string with following property keys: KUSTO_CLUSTER_URL, KUSTO_DATABASE_NAME, SERVICE_CLIENT_ID, SERVICE_CLIENT_SECRET, TENANT_ID. We do not want to store any other property key in config.

Always return the updated values by logically combining information from the user's input with the existing information to the current values. Only exclude information already given to you in the current values when the user specifically instructs to do so.
"""
    def _get_user_prompt(self):
        return """
The user wants to interact with the KQL expert chatbot which has the schema [SCHEMA] and config [CONFIG]. 

To process a user utterance [U], respond to the user based on conversational history of utterances from the user,  using information only in [SCHEMA].

The chatbot definition is as follows:

[SCHEMA]
{SCHEMA}

[CONFIG]
{CONFIG}

The following is the chat history. Messages from the bot are denoted by 'Bot:' and messages from the user are denoted by 'User:'. Based on the user's last input, please respond as described below.

Message History:
{chat_history}

The user's last input is: {user_input}

Based on the user utterance and context, generate the following: 

1. Response to be given to the user.

If a user utterance is not supported by the schemas present in [SCHEMA], respond back saying that you are unable to process the utterance and inform them about the kind of user utterances you are able to process.

If user utterance is to display the config, respond back by providing the [CONFIG] value. Please make sure that you mask or not display the service client secret.

Based on the above chatbot definition and the user's input, output the response to include the current state of the chatbot. Please print the output in the below format. Always print the [Response] section in the output without fail.

[Response]
<contents of response>
"""



class CommandChooser(AzureChatOpenAITool):
    def _get_system_prompt(self):
        return """
You are a KQL expert that can write queries based on tables schemas present in schema [SCHEMA] and can display the current configuration present in config [CONFIG]. 

The schema is a set of key-value pairs, where the keys are table names (which you can think of as KQL tables), values are schema of that table. We want the schema section names to be distinct from each other. 

The config is a json string with following property keys: KUSTO_CLUSTER_URL, KUSTO_DATABASE_NAME, SERVICE_CLIENT_ID, SERVICE_CLIENT_SECRET, TENANT_ID. We do not want to store any other property key in config.

Always return the updated values by logically combining information from the user's input with the existing information to the current values. Only exclude information already given to you in the current values when the user specifically instructs to do so.
"""

    def _get_user_prompt(self):
        return """
A user has given the following instruction to change the logic of the chatbot. To process a user utterance [U] first decide: 

Does the utterance require us to update config or configuration? If yes, then invoke “Update CONFIG (defined below) with the part [U-CONFIG] of the utterance [U] that is relevant to updating the [CONFIG]. 

“Update CONFIG", with part of the utterance [U-CONFIG] is done as follows: remove the [SCHEMA] section. Split the utterance [U-CONFIG]  into sentences. For each sentence [S], if [S] corresponds to setting the kusto cluster url then update KUSTO_CLUSTER_URL property in [CONFIG], if [S] corresponds to setting the database name then update KUSTO_DATABASE_NAME property in [CONFIG], if [S] corresponds to setting the service client id then update SERVICE_CLIENT_ID property in [CONFIG], if [S] corresponds to setting the tenant id then update TENANT_ID property in [CONFIG], if [S] corresponds to setting the service client secrete then update SERVICE_CLIENT_SECRET property in [CONFIG].

If user asks to display current config. Display current [CONFIG].

If the user asks any thing else other than the above mentioned kind of utterances, respond back saying that you are unable to process the utterance, and inform them about the kind of user utterances you are able to process.  

The current value of the config, [CONFIG] is below. Modify the below [CONFIG] to include changes if "Update CONFIG" is required. Make sure that the entire json is written in a single line. If no changes are required, please return the current value of [SCHEMA] without any change. Also make sure not to assume values for any of the properties whose values are not provided by user.
{CONFIG}

Please ensure to retain all the information in the above values, while only making modifications and additions to incorporate the user's input. Only exclude contents from the current values if the user specifically instructs to.
The user's instruction is: {user_input}

Based on the above description and the user's instruction, output the updated values of [SCHEMA]. Further, please output a one-line summary of the sections changed, if any. Please print the summary and updated values in the below format:

summary: <one-line summary of changes>

[CONFIG]
<contents of config>
    """
