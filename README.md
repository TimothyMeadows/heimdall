# heimdall.py
Heimdall AI Camera PoC using Intel® Movidius™ Neural Compute Stick.

This is a proof of concept that uses mobile net with a single movidius to process classifications with alerting from a webcam, or Raspberry PI 3.

![Chair Detection Example](https://github.com/TimothyMeadows/heimdall/raw/master/example/chairs.gif)

This script requires an Intel Movidius 1 with the [ncsdk](https://github.com/movidius/ncsdk) installed. Note this should be version 1 of the sdk and not 2.

- -s, --source ```Index of the video device. ex. 0 for /dev/video0```
- -pi, --pi ```Enable raspberry pi cam support. Only use this if your sure you need it.```
- -m, --match_threshold ```Percentage required for a mobile net object detection match in range of (0.0 - 1.0).```
- -g, --mobile_net_graph ```Path to the mobile net neural network graph file.```
- -l, --mobile_net_labels ```Path to labels file for mobile net.```
- -fps, --fps ```Show fps your getting after processing.```
- -alerts, --alerts ```Classification list that triggers alerts.```
- -email, --email ```Email address to send email alerts too.```
- -email_server, --email_server ```Email server to send email alerts from.```
- -email_username, --email_username ```Email server username to send email alerts with.```
- -email_password, --email_password ```Email server password to send email alerts with.```