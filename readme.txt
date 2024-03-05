Installation:

1. Python packages:

To install all required packages please head to the parent directory, where the requirements.txt is.

Run: pip install -r requirements.txt

Make sure all packages have installed successfully

2. Erlang

Head to https://www.erlang.org/downloads in order to download Erlang. On the download site there is a possibility to download an installer.

For other installation methods the following are available:

For Homebrew on macOS: brew install erlang
For MacPorts on macOS: port install erlang
For Ubuntu and Debian: apt-get install erlang
For Fedora: yum install erlang
For ArchLinux and Manjaro: pacman -S erlang
For FreeBSD: pkg install erlang

3. RabbitMQ
RabbitMQ requires to have Erlang installed first.

Head to https://www.rabbitmq.com/install-windows.html#downloads for Windows distribution and download the installer. Next proceed with the installation process.

RabbitMQ supports installation via Chocolatey choco install rabbitmq

For more information head to RabbitMQ repository - https://github.com/rabbitmq/rabbitmq-server/releases

At the end of installer process there will be a box asking if it should start the RabbitMQ process. Please mark it for the process to start. If this option will not be selected, the RabbitMQ has to be manually started by finding in the Windows search bar RabbitMQ Service - start or by heading to the RabbitMQ installation directory and running the script from C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.12\sbin.

After successful installation in order to start the application head to App/ directory and run python main.py. With correct set up the application should start with the following logs:

Now head to the provided url (in this case http://127.0.0.1:5000).


Instruction:  
After opening the application, click on the search icon located in the upper right corner of the page to select the company ticker.  
You will then be redirected promptly to the dashboard, where various details about the chosen stock are presented, including graphs showing stock price and sentiment values.  
Below these, you will find a list of articles.  

Above the graph on the left, next to the label 'For,' you can select the number of days into the future for which you would like to make predictions (the default value is 'Tomorrow').  
Similarly, above the graph on the left, next to the label 'From,' you can choose the range of past days for which historical price data is provided.  

Below the graph on the left, you have the option to choose an available model or create your own custom model.  
This can be done by selecting the number of epochs, layers, and neurons, deciding whether to use NLP data, and choosing the model architecture (N-BEATS or the default LSTM).  
After clicking 'Apply,' the predictions will be displayed on the graph once the model is ready.  

Above the graph on the right, next to the label 'From,' you can select the number of days in the past for which historical sentiment data is provided.  

Below the graphs, there is a dynamic list of articles, presenting the latest news on the stock.  
