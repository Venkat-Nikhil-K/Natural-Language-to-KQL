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
        return [KBRep(),LogicRep(),VariablesRep(),VariableValuesRep()]

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
            kb = ''
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
                        kb += '\n' + await self.process_document_with_llm(fileData)
            self._project.representations["kb"].text += '\n' + kb
            await self._progress(Response(
                type="output",
                message="Ingested the data and updated the knowledge base!",
                project=self._project
            ))
            return "Ingested the data and updated the knowledge base!"

        else:
            tool = CommandChooser(credentials=self._credentials)
            cmd = await tool.run(
                KB=self._project.representations["kb"].text,
                Logic=self._project.representations["logic"].text,
                Variables=self._project.representations["variables"].text,
                user_input=text
            )

            change_summary, kb, logic, variables = self.process_output(cmd)
            print(change_summary, kb, logic, variables)
            self._project.representations["kb"].text = kb
            self._project.representations["logic"].text = logic
            self._project.representations["variables"].text = variables

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
            KB=self._project.representations["kb"].text,
            Logic=self._project.representations["logic"].text,
            Variables=self._project.representations["variables"].text,
            chat_history=chat_history,
            Variable_values=self._project.representations["variable_values"].text,
            user_input=text)
        
        response_index = -1
        variables_index = -1
        for idx, line in enumerate(output):
            if '[Response]' in line or 'Response:' in line:
                response_index = idx
            if '[Variable_values]' in line or 'Variable_values:' in line:
                variables_index = idx
        
        if variables_index == -1 and response_index == -1:
            response = output
        else:
            response = '\n'.join(output[response_index + 1:variables_index])
            variables = '\n'.join(output[variables_index + 1:])
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
                message='KB modified',
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
            if '[KB]' in line or 'KB:' in line:
                kb_index = idx
            if '[Logic]' in line or 'Logic:' in line:
                logic_index = idx
            if '[Variables]' in line or 'Variables:' in line:
                variables_index = idx

        kb = '\n'.join(output[kb_index + 1:logic_index])
        logic = '\n'.join(output[logic_index + 1:variables_index])
        variables = '\n'.join(output[variables_index + 1:])
        return change_summary, kb, logic, variables
    
class KBRep(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="kb",
            text="",
            type="md"
        )

class LogicRep(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="logic",
            text="",
            type="md"
        )

class VariablesRep(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="variables",
            text="",
            type="md"
        )

class VariableValuesRep(PbyCRepresentation):
    def _get_initial_values(self):
        return Representation(
            name="variable_values",
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
You are a chatbot designed to chat with users, defined by a knowledge base [KB] and a set of logical rules [Logic].  The logic uses variables defined in [Variables]. 

The knowledge base is a set of key-value pairs, where the keys are section names (which you can think of as topic names), values are information on that topic. Initially the knowledge base is empty. We want the knowledge base section names to be distinct from each other. The set of logical rules [Logic] is a set of rules governing the behavior of the bot. Initially [Logic] is empty. We want to ensure that the set of rules don't contradict each other.

The set of logical rules [Logic] is a set of rules governing the behavior of the bot. Initially [Logic] is empty. We want to ensure that the set of rules don't contradict each other. 

[Variables] is a set of variables that are used to keep track of the state of engagement of the user with the bot. Initially [Variables] is also empty. Each variable should have an initial value, and a rule to update its value depending on the interaction between the user and the bot. 

The logic is a numbered list of instructions and rules for the chatbot. Variables contains a list of state variables, their initial state and their update conditons.

You will facilitate the user's interaction with the chatbot. Please ensure to always follow the given output format. Please ensure to always give the user a response, if you are not sure how to process their input please say so.

"""
    def _get_user_prompt(self):
        return """
The user wants to interact with the chatbot defined by defined by the knowledge base [KB], the logic [Logic], and variable definitions [Variables]. 

To process a user utterance [U], respond to the user based on conversational history of utterances from the user,  using information only in [KB], governed only by the rules in [Logic].  If an utterance is not admitted by the rules in [Logic], do not process it. When generating responses, ensure that all rules in [Logic] are followed. Use the current value of [Variables] while enforcing the logic and rules.

The chatbot definition is as follows:

[KB]
{KB}

[Logic]
{Logic}

[Variables]
{Variables}

The following is the chat history. Messages from the bot are denoted by 'Bot:' and messages from the user are denoted by 'User:'. Based on the user's last input, please respond as described below.

Message History:
{chat_history}

The following are the current values of the variables. Note that initially, [Variables] == [Variable_values]. In this case, return just the updated values of the variables, and assign random values to variables if needed.

[Variable_values]
{Variable_values}

The user's last input is: {user_input}

Based on the user utterance and context, generate the following: 

1. Response to be given to the user. 

2. Update all [Variable_values] and output the names of the variables and values. Please note that if some of the variables require choosing random values, assign it any random value that fits the variable's description and return the randomly chosen value in the updated list.

If a user utterance is not supported by the rules in [Logic], respond back saying that you are unable to process the utterance and inform them about the kind of user utterances you are able to process. In this case, return the variables unchanged.

Based on the above chatbot definition and the user's input, output the response, updated value of [Variable_values] to include the current state of the chatbot. Please print the output and updated variables in the below format. Always print the [Response] and [Variable_values] section in the output without fail.

[Response]
<contents of response>

[Variable_values]
<values of variables>
"""



class CommandChooser(AzureChatOpenAITool):
    def _get_system_prompt(self):
        return """
You are a bot designed to develop a chatbot defined by a knowledge base [KB] and a set of logical rules [Logic].  The logic uses variables defined in [Variables]. 

The knowledge base is a set of key-value pairs, where the keys are section names (which you can think of as topic names), values are information on that topic. Initially the knowledge base is empty. We want the knowledge base section names to be distinct from each other. 

The set of logical rules [Logic] is a set of rules governing the behavior of the bot. Initially [Logic] is empty. We want to ensure that the set of rules don't contradict each other. 

[Variables] is a set of variables that are used to keep track of the state of engagement of the user with the bot. Initially [Variables] is also empty. Each variable should have an initial value, and a rule to update its value depending on the interaction between the user and the bot. 

The logic is a numbered list of instructions and rules for the chatbot. Variables contains a list of state variables, their initial state and their update conditons.

Users enter instructions that you need to process in order to help them develop their own chatbot. 

Always return the updated values by logically combining information from the user's input with the existing information to the current values. Only exclude information already given to you in the current values when the user specifically instructs to do so.
    """

    def _get_user_prompt(self):
        return """
A user has given the following instruction to change the logic of the chatbot. To process a user utterance [U] first decide: 

Does the utterance require us to update KB? If yes, then invoke “Update KB” (defined below) with the part [U-KB] of the utterance [U] that is relevant to updating the [KB]. 

Does the utterance require us to update Logic? If yes, then say “Update Logic” (defined below) with the part [U-Logic] of the utterance [U] that is relevant to updating the logical rules [Logic] 

Does the utterance require us to update the set of variables? If yes, then say “Update Variables” (defined below) with the part [U-Variables] of the utterance [U] that is relevant to updating  [Variables] 

Note that you can decide to invoke none, one or more of the updates  “Update KB”, “Update Logic” and “Update Variables”.  

“Update KB”, with the current [KB] and part of the utterance [U-KB] is done as follows: Split the utterance [U-KB]  into sentences. For each sentence [S], if [S] corresponds to a section that is already in the [KB], merely add the utterance to the value corresponding to that section. Otherwise, choose a new section name [N], and add the sentence [S] in the value corresponding to that section. 

“Update Logic”, with the current logic [Logic] and part of the utterance [U-Logic] is done as follows. Split the utterance [U-Logic] into sentences.  

For each sentence [S], first convert the sentence to a concise form referring to variables in [Variables] as needed. Then, if [S] corresponds to a rule [R] that is already part of the [Logic], then update [R], considering the intent behind sentence [S]. Otherwise, add [S] as a new rule to the [Logic]. Please only include rules and instructions given by the user.

“Update Variables” with the current set of variables [Variables] and part of the utterance [U-Variables] is done as follows. Split the utterance [U-Logic] into sentences.  

For each sentence [S], if [S] corresponds to variable [V] that is already part of [Variables], then update the initial value or the logic to update its value, taking into consideration the sentence [S]. Otherwise, create a new variable with a suitable name [V] and add update the initial value and logic to update its value, as specified by the sentence [S]. 

In addition, if the user asks you to show the contents of the [KB], [Logic] or [Variables], oblige them. 

If the user asks any thing else other than the above mentioned kind of utterances, respond back saying that you are unable to process the utterance, and inform them about the kind of user utterances you are able to process.  

The current value of the knowledge base, [KB] is below. Modify the below [KB] to include changes if "Update KB" is required. If no changes are required, please return the current value of [KB] without any change.
{KB}

The current value of the knowledge base, [Logic] is below. Modify the below [Logic] to include changes if "Update Logic" is required. If no changes are required, please return the current value of [Logic] without any change. 
{Logic}

The current value of the knowledge base, [Variables] is below. Modify the below [Variables] to include changes if "Update Variables" is required. If no changes are required, please return the current value of [Variables] without any change.
{Variables}

Please ensure to retain all the information in the above values, while only making modifications and additions to incorporate the user's input. Only exclude contents from the current values if the user specifically instructs to.
The user's instruction is: {user_input}

Based on the above description and the user's instruction, output the updated values of [KB], [Logic] and [Variables]. Further, please output a one-line summary of the sections changed, if any. Please print the summary and updated values in the below format:

summary: <one-line summary of changes>

[KB]
<contents of kb>

[Logic]
<contents of logic>

[Variables]
<contents of variables>
    """
