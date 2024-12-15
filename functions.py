import time
import streamlit as st
import re

def spinner(text:str,seconds:int,done_display_time: int = 2):
    """
        st.info('This is a purely informational message', icon="ℹ️")
        
        st.error('This is an error', icon="🚨")
        
        e = RuntimeError('This is an exception of type RuntimeError')
        st.exception(e)
        
    """
    with st.spinner(text):
        time.sleep(seconds)
    done_message = st.empty()
    done_message.success('Done!',icon="✅")
    time.sleep(done_display_time)
    done_message.empty()

def progress_bar(title:str):
    bar = st.progress(0,text=title)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        bar.progress(percent_complete + 1,text=title)
    time.sleep(1)
    bar.empty()
    completed("Completed")

def model_train():
    st.empty()
    with st.status("Model running...", expanded=True) as status:
        st.write("Model selecting...")
        time.sleep(1)
        st.write("Training...")
        time.sleep(2)
        st.write("Predicting...")
        time.sleep(1)
        status.update(label="Excution completed!",state = "complete",expanded=False)

## STATUS ELEMENTS - CALLOUTS

def warn(text:str,seconds:int,done_display_time: int = 2):
    done_message = st.empty()
    done_message.warning(text, icon="⚠️")
    time.sleep(seconds)
    time.sleep(done_display_time)
    done_message.empty()

def err(text:str,seconds:int,done_display_time: int = 2):
    done_message = st.empty()
    done_message.error(text, icon="❌")
    time.sleep(seconds)
    time.sleep(done_display_time)
    done_message.empty()
    
def inf(text:str,seconds:int,done_display_time: int = 2):
    done_message = st.empty()
    done_message.info(text, icon="ℹ️")
    time.sleep(seconds)
    time.sleep(done_display_time)
    done_message.empty()

def completed(text:str,done_display_time:int = 2):
    done_message = st.empty()
    done_message.success(text)
    time.sleep(done_display_time)
    done_message.empty()
    

# Validation

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    else:
        return False