from copy                        import deepcopy
from datetime                    import timedelta
from flask                       import Flask, request
from json                        import dumps
from os                          import getenv
from prometheus_api_client       import MetricSnapshotDataFrame, PrometheusConnect
from prometheus_api_client.utils import parse_datetime
from requests                    import get
from sklearn.preprocessing       import MinMaxScaler
from torch                       import load, tensor, zeros as torch_zeros
from torch.nn                    import Linear, LSTM, Module, MSELoss
from urllib3                     import disable_warnings

disable_warnings()

PROMETHEUS_URL   = getenv('PROMETHEUS_URL')
PROMETHEUS_TOKEN = getenv('PROMETHEUS_TOKEN')
NAMESPACE        = getenv('NAMESPACE')
TELEGRAM_TOKEN   = getenv('TELEGRAM_TOKEN')
TELEGRAM_API     = f'https://api.telegram.org/bot{ TELEGRAM_TOKEN }'


class PrometheusLSTM(Module):

    def __init__(self, input_size, hidden_size, stacked_layers):

        super().__init__()

        self.hidden_size    = hidden_size
        self.stacked_layers = stacked_layers

        self.LSTM   = LSTM(input_size, hidden_size, stacked_layers, batch_first = True)
        self.linear = Linear(hidden_size, 1)

    def forward(self, x):

        batch_size = x.size(0)

        h0 = torch_zeros(self.stacked_layers, batch_size, self.hidden_size).to('cpu')
        c0 = torch_zeros(self.stacked_layers, batch_size, self.hidden_size).to('cpu')

        output, _ = self.LSTM(x, (h0, c0))
        output    = self.linear(output[:, -1, :])

        return output


def transform_and_normalize(metric_data, lookback):

    metric_data = deepcopy(metric_data)
    metric_data = metric_data[['timestamp', 'value']]

    for index in range(1, lookback + 1):

        metric_data[f't - {index}'] = metric_data['value'].shift(index)

    metric_data.set_index('timestamp', inplace = True)
    metric_data.dropna(inplace = True)

    return metric_data


prometheus_connect = PrometheusConnect(
    url         = PROMETHEUS_URL,
    headers     = { 'Authorization' : f'bearer { PROMETHEUS_TOKEN }' },
    disable_ssl = True
)

metric_name   = 'pod:container_cpu_usage:sum'
start_time    = parse_datetime('1h')
end_time      = parse_datetime('now')
chunk_size    = timedelta(minutes = 1)
lookback      = 4
scaler        = MinMaxScaler(feature_range = (-1, 1))
loss_function = MSELoss()

label_config = {
    'prometheus' : 'openshift-monitoring/k8s',
    'namespace'  : NAMESPACE
}

model = PrometheusLSTM(1, 4, 1)
model.to('cpu')
model.load_state_dict(load('model.pt', weights_only = True))
model.eval()

app = Flask(__name__)


@app.route('/', methods = ['GET'])
def predict() -> None:

    metric_data = prometheus_connect.get_metric_range_data(
        metric_name  = metric_name,
        label_config = label_config,
        start_time   = start_time,
        end_time     = end_time,
        chunk_size   = chunk_size
    )

    metric_data = MetricSnapshotDataFrame(metric_data)
    metric_data = transform_and_normalize(metric_data, lookback)
    metric_data = metric_data.to_numpy()
    metric_data = scaler.fit_transform(metric_data)
    metric_data = metric_data[:, 1:]
    metric_data = metric_data.reshape((-1, lookback, 1))
    metric_data = tensor(metric_data).float()
    metric_data = metric_data.to('cpu')

    predicted = model(metric_data)
    loss      = loss_function(predicted, metric_data[:, 0])

    print(loss)