import pandas as pandas
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import numpy as np

# Load the data set
dataUncleaned = pandas.read_csv("Data for ML/NF-UNSW-NB15-v2.csv")

# Encode Data
attackCatagories = [['Benign', 'Analysis', 'Backdoor', 'Fuzzers', 'DoS', 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']]

ordinalColumns = ["Attack"]

encoding = make_column_transformer((OrdinalEncoder(categories=attackCatagories),ordinalColumns))
transformedData = encoding.fit_transform(dataUncleaned)

# Get Unencoded Data
unchangedColumns = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE"]
unchangedData = dataUncleaned[unchangedColumns].values

# Concatenate Data
dataCleaned = np.concatenate((transformedData, unchangedData), axis=1)

# Create Collumn Names
dataCleanedColumns = unchangedColumns + ordinalColumns

# Created a Data Frame
cleanedDataFrame = pandas.DataFrame(dataCleaned, columns=dataCleanedColumns)

# Normalise Collumns
columnsToNormalise = [col for col in cleanedDataFrame.columns if col != 'Label' and col != 'IPV4_SRC_ADDR' and col != 'IPV4_DST_ADDR' and col != 'L4_SRC_PORT' and col != 'L4_DST_PORT']
scaler = MinMaxScaler()
cleanedDataFrame[columnsToNormalise] = scaler.fit_transform(cleanedDataFrame[columnsToNormalise])

cleanedDataFrame.to_csv('Data for ML/cleanedData.csv')
print("Data Cleaned")