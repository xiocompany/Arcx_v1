import websocket
import json
import pandas as pd
from datetime import datetime
import os
import time

# Global buffer for WebSocket message fragments
message_buffer = ""

# Global WebSocket reference to manage connection
global_ws = None

def process_payload(payload, csv_filename="chart_data.csv"):
    global global_ws
    try:
        data = json.loads(payload)

        if data.get("m") == "timescale_update":
            print("Processing timescale_update message.")
            timescale_data = data.get("p", [])[1].get("sds_1", {}).get("s", [])
            if not timescale_data or not isinstance(timescale_data, list):
                print("No chart data found in 's'. Exiting.")
                return

            chart_data = []
            for entry in timescale_data:
                if not isinstance(entry.get("v", []), list) or len(entry["v"]) < 6:
                    print(f"Invalid 'v' data in entry: {entry}")
                    continue

                time_val = datetime.utcfromtimestamp(entry["v"][0]).strftime('%Y-%m-%d %H:%M:%S')
                open_price = entry["v"][1]
                high_price = entry["v"][2]
                low_price = entry["v"][3]
                close_price = entry["v"][4]
                volume = entry["v"][5]
                chart_data.append([time_val, open_price, high_price, low_price, close_price, volume])

            if not chart_data:
                print("No valid chart data to save. Exiting.")
                return

            df = pd.DataFrame(chart_data, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

            file_exists = os.path.isfile(csv_filename)
            if file_exists:
                df.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Chart data appended to {csv_filename}")
            else:
                df.to_csv(csv_filename, index=False)
                print(f"Chart data saved to {csv_filename}")

            if global_ws:
                print("Closing WebSocket connection.")
                global_ws.close()
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}. Payload context: {payload}")
    except KeyError as e:
        print(f"Key error: {e}. Check the structure of the payload.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def on_message(ws, message):
    global message_buffer, global_ws
    global_ws = ws

    message_buffer += message

    while "~m~" in message_buffer:
        try:
            parts = message_buffer.split("~m~", 2)
            if len(parts) < 3:
                break
            length = int(parts[1])
            if len(parts[2]) < length:
                break
            payload = parts[2][:length]
            message_buffer = parts[2][length:]

            if '"m":"timescale_update"' in payload:
                print("Processing complete payload.")
                process_payload(payload)

        except ValueError as e:
            print(f"Value error while processing message: {e}")
            break

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    global global_ws
    global_ws = ws

    print("WebSocket connection opened")

    auth_message = {
        "m": "set_auth_token",
        "p": ["eyJhbGciOiJSUzUxMiIsImtpZCI6IkdaeFUiLCJ0eXAiOiJKV1QifQ.eyJ1c2VyX2lkIjo5NTE3NDMyLCJleHAiOjE3MzYxOTYxMjgsImlhdCI6MTczNjE4MTcyOCwicGxhbiI6IiIsImV4dF9ob3VycyI6MSwicGVybSI6IiIsInN0dWR5X3Blcm0iOiIiLCJtYXhfc3R1ZGllcyI6MiwibWF4X2Z1bmRhbWVudGFscyI6MSwibWF4X2NoYXJ0cyI6MSwibWF4X2FjdGl2ZV9hbGVydHMiOjEsIm1heF9zdHVkeV9vbl9zdHVkeSI6MSwiZmllbGRzX3Blcm1pc3Npb25zIjpbXSwibWF4X292ZXJhbGxfYWxlcnRzIjoyMDAwLCJtYXhfYWN0aXZlX3ByaW1pdGl2ZV9hbGVydHMiOjUsIm1heF9hY3RpdmVfY29tcGxleF9hbGVydHMiOjEsIm1heF9jb25uZWN0aW9ucyI6Mn0.nTs1cN2H4bm_f2dF9E5q42BXIXpe50k_hVc30x_HYS08ApwyNkPGgV4FT49MIqfULSsgIGu-zGswwej6Qmv-C0b3rer88vclWcKiXIZrsg3BnWtq8uh1RbCUfDLu1BUmVnGHvoUtZTjal3XzQsV-XjqNZ73uqr-bI91CLPlVM2g"]
    }
    ws.send(encode_message(auth_message))

    locale_message = {
        "m": "set_locale",
        "p": ["en", "US"]
    }
    ws.send(encode_message(locale_message))

    chart_session_message = {
        "m": "chart_create_session",
        "p": ["cs_12345", ""]
    }
    ws.send(encode_message(chart_session_message))

    quote_session_message = {
        "m": "quote_create_session",
        "p": ["qs_12345"]
    }
    ws.send(encode_message(quote_session_message))

    resolve_symbol_message = {
        "m": "resolve_symbol",
        "p": [
            "cs_12345",
            "sds_sym_1",
            "={\"adjustment\":\"splits\",\"currency-id\":\"XTVCUSDT\",\"session\":\"regular\",\"symbol\":\"BINANCE:BTCUSDT\"}"
        ]
    }
    ws.send(encode_message(resolve_symbol_message))

    create_series_message = {
        "m": "create_series",
        "p": ["cs_12345", "sds_1", "s1", "sds_sym_1", "1", 300, ""]
    }
    ws.send(encode_message(create_series_message))

def encode_message(message):
    message_str = json.dumps(message)
    return f"~m~{len(message_str)}~m~{message_str}"

# WebSocket URL
url = "wss://data.tradingview.com/socket.io/websocket"

# Loop to rerun every 60 seconds
while True:
    print("Starting WebSocket session...")
    ws = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()
    print("WebSocket session ended. Waiting 60 seconds before reconnecting...")
    time.sleep(60)
