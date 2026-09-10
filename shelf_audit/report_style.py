"""Shared plain styling for standalone HTML reports."""
import re
from pathlib import Path

CSS = '''
:root{color-scheme:light!important;background:white!important;color:black!important}
*{box-sizing:border-box}
body{font-family:"Times New Roman",Times,serif!important;background:white!important;color:black!important;margin:20px auto!important;padding:0 12px!important;max-width:1000px!important}
h1{font-size:32px!important}h2{font-size:22px!important}
p,small,footer{color:black!important;line-height:1.45!important}
article,.panel,.card,li{background:white!important;color:black!important;border-radius:0!important;box-shadow:none!important}
article,section.panel{border:1px solid #777!important}
img,video,iframe,input,button,.card,.panel,.badge{border-radius:0!important;box-shadow:none!important}
a{color:#0000ee!important;text-decoration:underline!important}a:visited{color:#551a8b!important}
button,input{font-family:"Times New Roman",Times,serif!important}
button{background:#e8e8e8!important;color:black!important;border:2px outset #aaa!important}
button:active{border-style:inset!important}button:disabled{color:#666!important;cursor:wait}
input,select,textarea{background:white!important;color:black!important;border:1px solid #777!important;border-radius:0!important}
input[type=file]{border:0!important}
input::file-selector-button{font:inherit;background:#e8e8e8;color:black;border:2px outset #aaa;border-radius:0}
.badge,.eyebrow,.score,.muted,#status{color:black!important;letter-spacing:normal!important}
.error{color:#a00000!important}
#preview,.match img{background:#f4f4f4!important;border:1px solid #aaa!important}
a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid #000080;outline-offset:3px}
video{display:block;margin:16px 0;max-width:100%;background:black}
figure{margin:12px 8px}figcaption{font-size:15px;line-height:1.35}
header,footer{border-color:#777!important}
@media(max-width:640px){body{margin:12px auto!important;padding:0 8px!important}.grid,.workspace{grid-template-columns:1fr!important}h1{font-size:28px!important}}
table{border-collapse:collapse}th,td{border:1px solid #777;padding:6px}
'''


def apply_file(path):
    path=Path(path)
    text=path.read_text(encoding='utf-8')
    text=re.sub(r'<style id="plain-report-theme">.*?</style>', '', text, flags=re.S)
    block='<style id="plain-report-theme">'+CSS+'</style>'
    position=text.lower().rfind('</style>')
    if position>=0:
        position+=len('</style>');text=text[:position]+block+text[position:]
    else:
        text=block+text
    path.write_text(text,encoding='utf-8')
