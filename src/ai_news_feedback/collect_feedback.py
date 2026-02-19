import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

load_dotenv('.env', override=True)

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
TARGET_CHANNEL = os.getenv('TARGET_CHANNEL')

client = WebClient(token=SLACK_BOT_TOKEN)

def fetch_messages(channel, limit=100):
    try:
        result = client.conversations_history(channel=channel, limit=limit)
        return result['messages']
    except SlackApiError as e:
        print(f"Error fetching messages: {e}")
        return []

def fetch_replies(channel, thread_ts):
    try:
        result = client.conversations_replies(channel=channel, ts=thread_ts)
        return result['messages'][1:]  # thread root msg is first, so skip
    except SlackApiError:
        return []

def main():
    # チャンネルID取得
    try:
        channels = client.conversations_list(types="public_channel,private_channel")["channels"]
        cid = None
        for c in channels:
            if c.get('name') == TARGET_CHANNEL.lstrip('#'):
                cid = c['id']
                break
        if not cid:
            print(f"No such channel: {TARGET_CHANNEL}")
            return
    except SlackApiError as e:
        print(f"Error fetching channel list: {e}")
        return

    messages = fetch_messages(cid)
    for msg in messages:
        out = {
            'ts': msg.get('ts'),
            'user': msg.get('user'),
            'text': msg.get('text'),
            'reactions': msg.get('reactions', []),
            'reply_count': msg.get('reply_count', 0),
            'pinned_to': msg.get('pinned_to', []),
        }
        if out['reply_count'] > 0:
            replies = fetch_replies(cid, msg['ts'])
            out['replies'] = [
                {'user': r.get('user'), 'text': r.get('text'), 'ts': r.get('ts')}
                for r in replies
            ]
        print(json.dumps(out, ensure_ascii=False))

if __name__ == '__main__':
    main()
