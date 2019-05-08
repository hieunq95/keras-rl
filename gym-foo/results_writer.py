import xlsxwriter

class MCMLWriter():
    """
    Writer for exporting results to xlsx file
    """
    def __init__(self, workbook):
        self.worksheet1 = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True})
        self.worksheet1.write('A1', 'Episode', bold)
        self.worksheet1.write('B1', 'Episode_steps', bold)
        self.worksheet1.write('C1', 'Total_reward', bold)
        self.worksheet1.write('D1', 'Mean_reward', bold)
        self.worksheet1.write('E1', 'Energy', bold)
        self.worksheet1.write('F1', 'Latency', bold)
        self.worksheet1.write('G1', 'Training_data', bold)
        self.worksheet1.write('H1', 'Training_data_mean', bold)

        self.worksheet2 = workbook.add_worksheet()
        self.worksheet2.write('A1', 'Episode', bold)
        self.worksheet2.write('B1', 'Epsilon', bold)

    def general_write(self, logs, episode):
        """
        Write information to xlsx file

        :param logs: General info

        :param episode: Episode

        :return:
        """
        info = dict(*logs)
        self.worksheet1.write(episode, 0, episode)
        self.worksheet1.write(episode, 1, info.get('episode_steps'))
        self.worksheet1.write(episode, 2, info.get('episode_reward'))
        self.worksheet1.write(episode, 3, info.get('reward_mean'))
        self.worksheet1.write(episode, 4, info.get('energy'))
        self.worksheet1.write(episode, 5, info.get('latency'))
        self.worksheet1.write(episode, 6, info.get('training_data'))
        self.worksheet1.write(episode, 7, info.get('training_data_mean'))

    def epsilon_write(self, logs, episode):
        """
        Write epsilon (greedy) to xlsx file
        :param logs: Current epsilon
        :param episode: Episode
        :return:
        """
        epsilon = logs
        self.worksheet2.write(episode, 0, episode)
        self.worksheet2.write(episode, 1, epsilon)

    def print_test(self, info):
        info = dict(*info)
        print("--------------------------------------")
        print("step_total {}, episode {}, episode_steps {}, episode_reward {}, mean_reward {}, "
              "energy {}, latency {}".
              format(info.get('step_total'), episode, info.get('episode_steps'), info.get('episode_reward'),
                     info.get('reward_mean'), info.get('energy'), info.get('latency')))
