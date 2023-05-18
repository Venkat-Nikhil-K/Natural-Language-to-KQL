from typing import List
from pbyc.engines import PbyCEngine
from pbyc.representations import PbyCRepresentation
from pbyc.tools.aoi_chat import AzureChatOpenAITool
from pbyc.tools.file import FileTool
from pbyc.tools import PbyCTool
from pbyc.types import Representation
from pbyc.types import ChangedRepresentation, Representation, Response, ChatMessage
import aiohttp
import tiktoken

class KqlExpertBotTool(PbyCEngine):
    def _get_representations(self):
        return [TableSchema(),Configuration()]

    def get_tokensize(self, text):
        enc = tiktoken.encoding_for_model('text-davinci-003')
        return len(enc.encode(text))
    
    async def _take_utterance(self, text:str, **kwargs):
        files = kwargs.get("files", [])
        if files and len(files) > 0:
            await self._progress(
                Response(
                    type="thought",
                    message="Processing uploaded files",
                    project=self._project
                ))
            TableSchema = ''
            for file in files:
                async with aiohttp.ClientSession() as session:
                    async with session.get(file.url) as response:
                        filepath = '/tmp/' + file.name
                        if response.status == 200:
                            with open(filepath, 'wb') as f:
                                while True:
                                    chunk = await response.content.read(1024)  # Adjust chunk size as needed
                                    if not chunk:
                                        break
                                    f.write(chunk)
                            print(f"File downloaded and saved to {filepath}")
                        else:
                            raise Exception(f'File download response: {resp.status} - {resp.reason}')
                        fileTool = FileTool()
                        fileData = await fileTool.run(
                            file=filepath
                        )
                        TableSchema += '\n' + await self.process_document_with_llm(fileData)
            self._project.representations["TableSchema"].text += '\n' + TableSchema
            await self._progress(Response(
                type="output",
                message="Ingested the data and updated the TableSchema!",
                project=self._project
            ))
            return "Ingested the data and updated the TableSchema!"

        else:
            tool = CommandChooser(credentials=self._credentials)
            cmd = await tool.run(
            TableSchema=self._project.representations["TableSchema"].text,
            Configuration=self._project.representations["Configuration"].text,
            user_input=text
            )

        change_summary, kb, logic, variables = self.process_output(cmd)
        print(change_summary, kb, logic, variables)
        self._project.representations["TableSchema"].text = kb
        self._project.representations["Configuration"].text = logic

        await self._progress(
            Response(
                type="output",
                message=change_summary,
                project=self._project
            ))
        return change_summary
    
    async def _get_output(self, text:str, **kwargs):

        chat_history = kwargs.get("chat_history", [])

        tool = OutputBot(credentials=self._credentials)
        output = await tool.run(
            KB=self._project.representations["TableSchema"].text,
            Logic=self._project.representations["Configuration"].text,
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
                message='TableSchema modified',
                project=self._project
            ))

    async def process_document_with_llm(self, kb):
        """
        Converts knowledge from documents into a knowledge base sections.
        """
        re_kb = kb
        out_kb = ''
        textLLM = HtmlToKB(credentials=self._credentials)
        while self.get_tokensize(re_kb) > 3000:
            re_kbp = re_kb[:10000]
            re_kb = re_kb[10000:]
            out_kb += await textLLM.run(website_data=re_kbp)

        out_kb += await textLLM.run(website_data=re_kb)
        kb = out_kb 
        return kb

    def process_output(self, output):
        output = output.split('\n')
        for idx, line in enumerate(output):
            if 'summary' in line:
                change_summary = line.split(':')[1].strip()
            if '[TableSchema]' in line or 'TableSchema:' in line:
                TableSchemas_index = idx
            if '[Configuration]' in line or 'Configuration:' in line:
                Configuration_index = idx

        TableSchema = '\n'.join(output[TableSchemas_index + 1:TableSchemas_index])
        Configuration = '\n'.join(output[Configuration_index + 1:Configuration_index])
        return change_summary, TableSchema, Configuration
    
class TableSchema(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="TableSchema",
            text="",
            type="md"
        )

class Configuration(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="Configuration",
            text="",
            type="md"
        )

class HtmlToKB(AzureChatOpenAITool):
    def _get_system_prompt(self):
        return """
A user has just added new information from documents that need to be added to the knowledge base. 
- Please follow the given output format. All sections are key-value pairs, where they keys are section names and the values are the contents of the section.
"""
    def _get_user_prompt(self):
        return """
{website_data}
---
The above are the details obtained from a website or a document. First decide different sections to categorize these details into. Ensure that each section covers a separate set of details and that there are no overlaps between section names and details covered. Then rewrite the details separated into different sections where each section has the following format:

<name of the section>: <one or more lines of section details>

Exhaustively cover all information given above, especially covering all technical information.
"""

class OutputBot(AzureChatOpenAITool):
    def _get_system_prompt(self):
        return """
You are a bot designed to develop a KQL expert chatbot which has a TableSchema [TableSchema]. 

The TableSchema is a set of key-value pairs, where the keys are table names (which you can think of as KQL tables), values are TableSchema of that table. We want the TableSchema section names to be distinct from each other. 

Always return the updated values by logically combining information from the user's input with the existing information to the current values. Only exclude information already given to you in the current values when the user specifically instructs to do so.
"""
    def _get_user_prompt(self):
        return """
The user wants to interact with the KQL expert chatbot which has the TableSchema [TableSchema]. 

To process a user utterance [U], respond to the user based on conversational history of utterances from the user,  using information only in [TableSchema].

The chatbot definition is as follows:

[TableSchema]
{TableSchema}

The following is the chat history. Messages from the bot are denoted by 'Bot:' and messages from the user are denoted by 'User:'. Based on the user's last input, please respond as described below.

Message History:
{chat_history}

The user's last input is: {user_input}

Based on the user utterance and context, generate the following: 

1. Response to be given to the user.

If a user utterance is not supported by the schemas present in [TableSchema], respond back saying that you are unable to process the utterance and inform them about the kind of user utterances you are able to process.

Based on the above chatbot definition and the user's input, output the response to include the current state of the chatbot. Please print the output in the below format. Always print the [Response] section in the output without fail.

[Response]
<contents of response>
"""



class CommandChooser(AzureChatOpenAITool):
    def _get_system_prompt(self):
        return """
You are a bot designed to develop a KQL expert chatbot which has a TableSchema [TableSchema]. 

The TableSchema is a set of key-value pairs, where the keys are table names (which you can think of as KQL tables), values are TableSchema of that table. We want the TableSchema section names to be distinct from each other. 

Always return the updated values by logically combining information from the user's input with the existing information to the current values. Only exclude information already given to you in the current values when the user specifically instructs to do so.
    """

    def _get_user_prompt(self):
        return """
A user has given the following instruction to change the logic of the chatbot. To process a user utterance [U] first decide: 

Does the utterance require us to update TableSchema? If yes, then invoke “Update TableSchema (defined below) with the part [U-TableSchema] of the utterance [U] that is relevant to updating the [TableSchema]. 

“Update SCHEMA”, with the current [TableSchema] and part of the utterance [U-TableSchema] is done as follows: Split the utterance [U-TableSchema]  into sentences. For each sentence [S], if [S] corresponds to a section that is already in the [TableSchema], merely update the value corresponding to that section with the utterance. Otherwise, choose a new section name [N], and add the sentence [S] in the value corresponding to that section. 

In addition, if the user asks you to show the contents of the [TableSchema] oblige them. 

If the user asks any thing else other than the above mentioned kind of utterances, respond back saying that you are unable to process the utterance, and inform them about the kind of user utterances you are able to process.  

The current value of the knowledge base, [TableSchema] is below. Modify the below [TableSchema] to include changes if "Update TableSchema" is required. If no changes are required, please return the current value of [TableSchema] without any change.
{TableSchema}

Please ensure to retain all the information in the above values, while only making modifications and additions to incorporate the user's input. Only exclude contents from the current values if the user specifically instructs to.
The user's instruction is: {user_input}

Based on the above description and the user's instruction, output the updated values of [TableSchema]. Further, please output a one-line summary of the sections changed, if any. Please print the summary and updated values in the below format:

summary: <one-line summary of changes>

[TableSchema]
<contents of TableSchema>
    """
