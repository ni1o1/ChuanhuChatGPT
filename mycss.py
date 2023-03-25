mycss = '''
.message-wrap {
    height:60vh;
    min-height: 400px
}
#chatbot .wrap {background-color:#f1f1f1}
.message { 
    color: black !important;
    border-radius:5px !important;
    position: relative !important;
    line-height:20px !important;
    }
.user {
    right: 50px;
}
.user::before {
    position: absolute;
    top: 10px;
    right: -10px;
    content: '';
    width: 0;
    height: 0;
    border-right: 5px solid transparent;
    border-bottom: 5px solid transparent;
    border-left: 5px solid #7beb67;
    border-top: 5px solid transparent;
}
.user::after {
    position: absolute;
    content: '';
    top: 0px;
    right: -50px;
    width: 35px;
    height: 35px;
    border-radius:3px !important;
    background-image:url("https://img0.baidu.com/it/u=3828378951,1675897767&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=500");
    background-repeat: round;
}
.bot {
    left: 50px;
}
.bot::before {
    position: absolute;
    top: 10px;
    left: -10px;
    content: '';
    width: 0;
    height: 0;
    border-right: 5px solid #FFF;
    border-bottom: 5px solid transparent;
    border-left: 5px solid transparent;
    border-top: 5px solid transparent;
}
.bot::after {
    position: absolute;
    content: '';
    top: 0px;
    left: -50px;
    width: 35px;
    height: 35px;
    border-radius:3px !important;
    background-image:url("https://raw.githubusercontent.com/ni1o1/ChuanhuChatGPT/main/image/avatar2.png");
    background-repeat: round;
}
code {
    display: inline;
    white-space: break-spaces;
    border-radius: 6px;
    margin: 0 2px 0 2px;
    padding: .2em .4em .1em .4em;
    color: crimson;
    background-color: antiquewhite;
}
pre code {
    display: block;
    white-space: break-spaces;
    background-color: hsla(0, 0%, 0%, 72%);
    border: solid 5px var(--color-border-primary) !important;
    border-radius: 10px;
    padding: 0 1.2rem 1.2rem;
    margin-top: 1em !important;
    color: #FFF;
    box-shadow: inset 0px 8px 16px hsla(0, 0%, 0%, .2)
}
'''