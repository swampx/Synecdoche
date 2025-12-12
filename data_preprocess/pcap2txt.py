import nfstream
import yaml



from pathlib import Path

def process_pcap_to_txt(pcap_path, output_dir, flow_counter, sample_limit, max_pkt_number, min_packet_number):
    try:
        # Use nfstream to read the pcap file with the custom plugin to extract packet details
        stream_reader = nfstream.NFStreamer(source=pcap_path, splt_analysis=max_pkt_number, max_nflows=sample_limit)

        # Ensure the output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for flow in stream_reader:
            # Only process IPv4 TCP flows
            if flow.ip_version != 4 or flow.protocol not in [6, 17]:
                continue

            # 排除 ICMP 和 IGMP协议 (这个检查其实已经被上面的protocol != 6过滤掉了)
            # if flow.protocol in [1, 2]:
            #     continue

            # Skip STUN flows if application_name is available

            if not hasattr(flow, 'application_name'):
                continue
            if 'TLS' not in flow.application_name and 'HTTP' not in flow.application_name:
                continue

            directions = []
            time_intervals = []
            lengths = []
            packet_count = 0
            error_length = False

            # 获取实际的包数量（splt特征的有效长度）
            actual_packet_count = 0
            for i in range(max_pkt_number):
                if hasattr(flow, 'splt_direction') and len(flow.splt_direction) > i and flow.splt_direction[i] != -1:
                    actual_packet_count += 1
                else:
                    break

            # 如果包数量不足最小要求，跳过这个流
            if actual_packet_count < min_packet_number:
                continue

            # 处理每个包的信息
            for i in range(actual_packet_count):
                # 处理方向信息：1(正向)保持1，0(反向)变成-1，-1(没有包)变成0
                if hasattr(flow, 'splt_direction') and len(flow.splt_direction) > i:
                    direction_val = flow.splt_direction[i]
                    if direction_val == 0:
                        direction = 1  # 正向
                    elif direction_val == 1:
                        direction = -1  # 反向
                    else:  # direction_val == -1
                        direction = 0  # 没有包
                else:
                    direction = 0
                directions.append(direction)

                # 处理时间间隔信息，将-1替换为0
                if hasattr(flow, 'splt_piat_ms') and len(flow.splt_piat_ms) > i:
                    time_interval = flow.splt_piat_ms[i] if flow.splt_piat_ms[i] != -1 else 0
                else:
                    time_interval = 0
                time_intervals.append(time_interval)

                # 处理包大小信息，将-1替换为0，并限制最大值为1514
                if hasattr(flow, 'splt_ps') and len(flow.splt_ps) > i:
                    length = flow.splt_ps[i] if flow.splt_ps[i] != -1 else 0
                    if length >= 1514:
                        length = 1514
                else:
                    length = 0
                lengths.append(length)

                packet_count += 1
            # 如果包数量足够且没有错误，保存到文件
            if packet_count >= min_packet_number and not error_length:
                # 确保所有列表都有max_pkt_number的长度，不足的用0填充
                while len(directions) < max_pkt_number:
                    directions.append(0)
                while len(time_intervals) < max_pkt_number:
                    time_intervals.append(0)
                while len(lengths) < max_pkt_number:
                    lengths.append(0)

                output_txt_path = output_dir / f"{flow_counter}_{packet_count}.txt"
                with open(output_txt_path, 'w') as f:
                    f.write(" ".join(map(str, time_intervals[:max_pkt_number])) + '\n')
                    f.write(" ".join(map(str, directions[:max_pkt_number])) + '\n')
                    f.write(" ".join(map(str, lengths[:max_pkt_number])) + '\n')
                    f.close()
                print(f"Saved: {output_txt_path}")

                flow_counter += 1

                # 检查是否达到样本限制
                if flow_counter > sample_limit:
                    return flow_counter

        return flow_counter

    except ValueError as e:
        print(f"Error processing {pcap_path}: {e}")
        return flow_counter
    except Exception as e:
        print(f"Unexpected error processing {pcap_path}: {e}")
        return flow_counter



# 处理整个数据集
def process_dataset(input_folder, output_folder, sample_limit, max_pkt_number, min_packet_number):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # 遍历所有类别文件夹
    for category_folder in input_path.iterdir():
        if category_folder.is_dir():

            category_output_path = output_path / category_folder.name
            category_output_path.mkdir(parents=True, exist_ok=True)

            # 初始化类别中的流计数器
            flow_counter = 0

            # 遍历每个pcap文件
            for pcap_file in category_folder.rglob("*.pcap"):
                # 直接在类别文件夹下创建输出txt文件
                flow_counter = process_pcap_to_txt(pcap_file, category_output_path, flow_counter, sample_limit, max_pkt_number, min_packet_number)
                if flow_counter > sample_limit:
                    break


if __name__ == "__main__":
    print("extract feature from pcap")
    config =yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    input_folder = config['dataset']['pcap_folder'] + config['dataset']['name']
    output_folder = config['dataset']['txt_folder'] + config['dataset']['name']  # 输出txt结果根目录
    sample_limit = config['preprocess']['sample_limit']
    max_pkt_number = config['preprocess']['max_pkt_number']
    min_pkt_number = config['preprocess']['min_pkt_number']
    process_dataset(input_folder, output_folder, sample_limit, max_pkt_number, min_pkt_number)