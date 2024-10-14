import utils
import streamlit as st
from streaming import StreamHandler
from langchain.prompts import PromptTemplate

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Context aware chatbot", page_icon="")
st.header('Context aware chatbot')

class ContextChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
    
    @st.cache_resource
    def setup_chain(_self):
        template = """You are a(n) {adjective} pirate having a friendly conversation
    
        Current conversation:
        {history}
        Human: {input}
        AI:
        """
        prompt_template = PromptTemplate(input_variables=["history", "input", "adjective"], template=template)

        memory = ConversationBufferMemory()
        chain = ConversationChain(llm=_self.llm,
                                  prompt=prompt_template.partial(adjective="nasty"),
                                  memory=memory, verbose=False)
        return chain
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("  "):
                st_cb = StreamHandler(st.empty())
                result = chain.invoke(
                    {"input":user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(ContextChatbot, user_query, response)

if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()
