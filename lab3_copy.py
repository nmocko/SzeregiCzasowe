import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt


def draw_figure(set1, set2, dates, min_sim, max_sim, c):
    dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
    plt.title('Similarity of opening and closing Bitcoin prices')
    plt.plot(dates, set1, label="open", color="blue", linewidth=0.5)
    plt.plot(dates, set2, label="close", color="green", linewidth=0.5)
    plt.plot(dates[min_sim[0]:min_sim[1]], set1[min_sim[0]:min_sim[1]], label="the best similarity", color="red", linewidth=1)
    plt.plot(dates[min_sim[0]:min_sim[1]], set2[min_sim[0]:min_sim[1]], color="red", linewidth=1)
    plt.plot(dates[max_sim[0]:max_sim[1]], set1[max_sim[0]:max_sim[1]], label="the weakest similarity", color="black", linewidth=1)
    plt.plot(dates[max_sim[0]:max_sim[1]], set2[max_sim[0]:max_sim[1]], color="black", linewidth=1)
    
    plt.xlabel('date')
    plt.ylabel('price in USD')
    plt.legend()
    plt.savefig(f'figure_{c}')
    plt.show()

class Lab:

    csv_file = None
    csv_file_path = ''
    n = 0      # Number of rows

    def __init__(self, file_path):
        self.csv_file_path = file_path
        self.csv_file = pd.read_csv(file_path)
        self.n = len(self.csv_file)
        pass

    def printCSV(self):
        if self.csv_file is None:
            print("Read file with 'readCSVfile' method")
        else:
            print(self.csv_file)
        return

    def choose_sets_lab3(self, p, k, choice, sets_power=0):
        match choice:
            case 1:
                set_name = "Original"
                set1 = self.csv_file['Open']
                set2 = self.csv_file['Close']

            case 2:

                set_name = f"Power{sets_power}"
                set1 = self.csv_file[f'OpenPower{sets_power}']

                # If sets transformed with given power weren't previously calculated
                if len(set1) == 0:
                    print(f"Calculating new set for power: {sets_power}")
                    self.Power_set(sets_power)
                    set1 = self.csv_file[f'OpenPower{sets_power}']
                    set2 = self.csv_file[f'ClosePower{sets_power}']
                else:
                    set2 = self.csv_file[f'ClosePower{sets_power}']

            case 3:
                set_name = "Detrending"

                set1 = self.csv_file['OpenDetr']
                set2 = self.csv_file['CloseDetr']

            case _:
                print("Please enter a number 1, 2 or 3")
                return
        subsets, subset_count = self.Minkowski_standard(p, k, set1, set2)

        print(f"subsets: {subsets}")
        self.lab3_saving_results(subsets, subset_count, set_name, p, k)

        # self.csv_file[f'Minkowski_{p}'] = subsets[0]
        # self.csv_file.to_csv(self.csv_file_path, index=False)

    def Minkowski_standard(self, p, k, set1, set2):
        # print("Calculating...")
        set_len = len(set1)
        m = set_len / k        # Saving results to the file

        subsets = k*[0]
        subsets_count = k*[0]

        for i in range(set_len):
            ind = math.floor(i/m)

            subsets[ind] += math.pow(abs(set1[i] - set2[i]), p)
            subsets_count[ind] += 1

        for i in range(k):
            subsets[i] = math.pow(subsets[i]/subsets_count[i], 1/p)

        # print(subsets)
        # print(subsets_count)
        # print("Calculations completed!")


        # Saving results (lab3)
        # self.lab3_saving_results(subsets, subsets_count, set_name, p, k, set1, set2)
        return subsets, subsets_count

    def lab4_zad1(self, width, p):
        k = math.floor((width/100)*self.n)
        result_euclidean = []
        print(f"k: {k}")
        print(f"self.n: {self.n}")

        for i in range(k):
            result_euclidean.append(0)

        for i in range(self.n - k):

            set1 = list(self.csv_file['Open'][i:i+k+1].copy())
            set2 = list(self.csv_file['Close'][i:i+k+1].copy())
            subsets, subsets_count = self.Minkowski_standard(p, 1, set1, set2)
            result_euclidean.append(subsets[0])

        self.csv_file[f'MovingWindow{width}'] = result_euclidean
        self.csv_file.to_csv(self.csv_file_path, index=False)

        date = self.csv_file['Date']
        draw_figure(result_euclidean, result_euclidean, date.to_numpy().flatten(),
        [0,0], [0,0], choice)

        return

    def lab4_zad2(self, p):
        width = [50, 100, 150]
        result_set = []
        print(f"self.n: {self.n}")

        for i in range(len(width)):
            print(f"k: {width}")

            k = width[i]
            for j in range(k):
                result_set.append(0)

            set1 = list(self.csv_file['Open'][k:].copy())
            set2 = list(self.csv_file['Close'][:-k].copy())

            subsets, subsets_count = self.Minkowski_standard(p, 1, set1, set2)
            result_set.append(subsets[0])
            set_name = f"Shift_{p}_{k}"

            # self.csv_file[f'MovingWindow_{p}_{width}'] = result_set
            # self.csv_file.to_csv(self.csv_file_path, index=False)
            self.lab3_saving_results(subsets, subsets_count, set_name, p, 1)
        return


    def lab4_zad3(self):
        k = 10

        f = open('results', 'a')
        f.write(f"Euclidean standard (p = 2) for sets:\n\n")
        f.write(f"-10%                0%                  +10%\n\n")

        r = self.n/10
        for i in range(10):

            # -10%
            shift = self.n//100
            set1 = list(self.csv_file['Open'][math.ceil(i * r):math.ceil(r * (i + 1) - shift)].copy())
            set2 = list(self.csv_file['Close'][math.ceil(i * r + shift):math.ceil(r * (i + 1))].copy())
            subsets, subsets_count = self.Minkowski_standard(2, 1, set1, set2)
            f.write(f"{subsets[0]:<20}")


            # 0%
            set1 = list(self.csv_file['Open'][math.ceil(i*r):math.ceil(r*(i+1))].copy())
            set2 = list(self.csv_file['Close'][math.ceil(i*r):math.ceil(r*(i+1))].copy())
            subsets, subsets_count = self.Minkowski_standard(2, 1, set1, set2)
            f.write(f"{subsets[0]:<20}")

            # + 10%
            set1 = list(self.csv_file['Open'][math.ceil(i * r + shift):math.ceil(r * (i + 1))].copy())
            set2 = list(self.csv_file['Close'][math.ceil(i * r):math.ceil(r * (i + 1) - shift)].copy())
            subsets, subsets_count = self.Minkowski_standard(2, 1, set1, set2)
            f.write(f"{subsets[0]:<20}\n")
        f.write('\n\n')


    # Saving results to the file (lab3)
    def lab3_saving_results(self, subsets, subsets_count, set_name, p, k, set1=None, set2=None):
        f = open('results', 'a')
        if p == 2:
            f.write(f"Euclidean standard (p = 2) for sets '{set_name}':\n\n")
        else:
            f.write(f"Minkowski standard with p = {p}, for sets '{set_name}':\n\n")

        max_ind = 0
        min_ind = 0
        max_val = 0
        min_val = math.inf
        for i in range(k):
            if subsets[i] > max_val:
                max_val = subsets[i]
                max_ind = i
            if subsets[i] < min_val:
                min_val = subsets[i]
                min_ind = i
            f.write(f"Subset {i}: {subsets[i]}\n")

        min_sum = sum(subsets_count[:min_ind])
        max_sum = sum(subsets_count[:max_ind])

        if min_ind == 0:
            min_begin_ind = 0
            min_end_ind = subsets_count[min_ind] - 1
        else:
            min_begin_ind = min_sum
            min_end_ind = min_sum + subsets_count[min_ind] - 1

        if max_ind == 0:
            max_begin_ind = 0
            max_end_ind = subsets_count[max_ind] - 1
        else:
            max_begin_ind = max_sum
            max_end_ind = max_sum + subsets_count[max_ind] - 1

        f.write(f"\nMin: Subset {min_ind}: {subsets[min_ind]} --> subset_indexes: [{min_begin_ind}, {min_end_ind}]\n"
                f"Max: Subset {max_ind}: {subsets[max_ind]} --> subset_indexes: [{max_begin_ind}, {max_end_ind}]\n"
                f"============================\n\n")
        f.close()

        # print("Drawing a figure\nClose window to continue!")
        # date = self.csv_file['Date']
        # draw_figure(set1.to_numpy().flatten(), set2.to_numpy().flatten(), date.to_numpy().flatten(),  [min_begin_ind, min_end_ind], [max_begin_ind, max_end_ind], choice)

    # Method creating a new pair of columns transformed by powering them with value of 'p'
    def Power_set(self, p):
        trsf_set1 = []
        trsf_set2 = []

        print("Creating new columns...")
        for i in range(self.n):
            trsf_set1.append(self.csv_file['Open'][i]**p)
            trsf_set2.append(self.csv_file['Close'][i]**p)

        self.csv_file[f'OpenPower{p}'] = trsf_set1
        self.csv_file[f'ClosePower{p}'] = trsf_set2
        self.csv_file.to_csv(self.csv_file_path, index=False)
        print(f"New columns created for p = {p}.")
        return

    # Method deleting the pair of existing columns transformed by powering them with value of 'p'
    def Column_del(self, columnName, p=0):
        if p != 0:
            del self.csv_file[f'OpenPower{p}']
            del self.csv_file[f'ClosePower{p}']
        if columnName is not None:
            del self.csv_file[columnName]
        self.csv_file.to_csv(self.csv_file_path, index=False)
        print(f"Deleted columns transformed with name: {columnName}")

    # Method creating a new pair of columns transformed by subtracting trend value from each row value
    def Detrending(self):

        detr_set1 = []  # Column to create and add, de-trended "Open" column
        detr_set2 = []  # Column to create and add, de-trended "Open" column

        x = np.array([(i+1) for i in range(self.n)]).reshape(-1, 1)
        y1 = self.csv_file['Open'].values
        y1 = np.array(y1)
        y2 = self.csv_file['Close'].values
        y2 = np.array(y2)

        reg1 = LinearRegression()  # create object for the class
        reg2 = LinearRegression()  # create object for the class

        reg1.fit(x, y1)  # perform linear regression
        reg2.fit(x, y2)  # perform linear regression

        a1 = reg1.coef_[0]
        b1 = reg1.intercept_
        a2 = reg2.coef_[0]
        b2 = reg2.intercept_

        for i in range(self.n):
            subtract_value1 = (i+1)*a1 + b1
            subtract_value2 = (i+1)*a2 + b2

            detr_set1.append(self.csv_file['Open'][i] - subtract_value1)
            detr_set2.append(self.csv_file['Close'][i] - subtract_value2)

        self.csv_file[f'OpenDetr'] = detr_set1
        self.csv_file[f'CloseDetr'] = detr_set2
        self.csv_file.to_csv(self.csv_file_path, index=False)

        return 0


    def arithmetic_mean(self, set1, set2, l):

        am = 0

        for i in range(l):
            am += set2[i] + set1[i]

        am /= 2 * l
        return am

    def increase_mean(self, set1, set2, l):
        am = 0

        for i in range(l):
            am += set2[i] - set1[i]

        am /= l
        return am

    def Pearson(self, set1, set2, l):
        am1 = 0
        am2 = 0
        for i in range(l):
            am1 += set1[i]
            am2 += set2[i]
        am1 /= l
        am2 /= l

        numerator = 0
        denominator1 = 0
        denominator2 = 0

        for i in range(l):
            numerator += (set1[i] - am1) * (set2[i] - am2)
            denominator1 += (set1[i] - am1) * (set1[i] - am1)
            denominator2 += (set2[i] - am2) * (set2[i] - am2)

        pearson = numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        return pearson

    def lab5_zad2(self, ranges):
        #  ranges = [[L1, L2], [L1, L2], ..., [L1, L2]]

        n = len(ranges)
        results = [[[0.0 for _ in range(4)] for _ in range(3)] for _ in range(n)]

        for i in range(n):

            W = ranges[i][1] - ranges[i][0]
            set1 = list(self.csv_file['Open'][ranges[i][0]: ranges[i][1]].copy())
            set2 = list(self.csv_file['Close'][ranges[i][0]: ranges[i][1]].copy())
            results[i][0][0] = self.arithmetic_mean(set1, set2, W)
            results[i][0][1] = self.increase_mean(set1, set2, W)
            a, b = self.Minkowski_standard(2, 1, set1, set2)
            results[i][0][2] = a[0]
            results[i][0][3] = self.Pearson(set1, set2, W)

            set1 = list(self.csv_file['Open'][ranges[i][0]: ranges[i][1]].copy())
            set2 = list(self.csv_file['Close'][ranges[i][0] + W: ranges[i][1] + W].copy())
            results[i][1][0] = self.arithmetic_mean(set1, set2, W)
            results[i][1][1] = self.increase_mean(set1, set2, W)
            a, b = self.Minkowski_standard(2, 1, set1, set2)
            results[i][1][2] = a[0]
            results[i][1][3] = self.Pearson(set1, set2, W)

            set1 = list(self.csv_file['Open'][ranges[i][0] + W: ranges[i][1] + W].copy())
            set2 = list(self.csv_file['Close'][ranges[i][0]: ranges[i][1]].copy())
            results[i][2][0] = self.arithmetic_mean(set1, set2, W)
            results[i][2][1] = self.increase_mean(set1, set2, W)
            a, b = self.Minkowski_standard(2, 1, set1, set2)
            results[i][2][2] = a[0]
            results[i][2][3] = self.Pearson(set1, set2, W)

        return results

    def lab5_zad3(self, results, ranges):

        n = len(results)
        f = open('results', 'a')
        f.write('\n\n============== Lab5 ==========\n\n')
        f.write("1st column - arithmetic mean\n"
                "2nd column - increase mean\n"
                "3rd column - euclidean distance\n"
                "4th column - Pearson product-moment correlation coefficient\n\n")

        for p in range(3):
            if p == 0:
                f.write("concurrently\n")
            elif p == 1:
                f.write("second set W shifted\n")
            elif p == 2:
                f.write("first set W shifted\n")
            for i in range(1, n):
                j = 0
                comment = ""
                for k in range(4):
                    if abs(results[i-1][p][k]/10) < abs(results[i][p][k] - results[i-1][p][k]):
                        j += 1
                        comment += f'{k+1}; '
                if j > 1:
                    f.write(f'range {i}: {ranges[i][0]} - {ranges[i][1]}\n')
                    f.write(f'{results[i][0]} <=== {j} anomalies in columns {comment}\n')
            f.write("\n")

        f.close()


if __name__ == "__main__":

    power = 2
    sets = 1
    choice = 1  # Choose a set:  1. Original  2. Power  3. Detrending

    # Which transformed-with-power sets should be used
    sets_power = 2

    lab = Lab('BTC-USD_changed.csv')

    # Create a new pair of columns transformed with power "p"
    # lab.Power_set(p)

    # Delete an existing pair of columns transformed with power "p"
    # lab.Column_del("MovingWindow1")
    # lab.Column_del("MovingWindow5")

    # Create detrending columns of the original columns
    # lab.Detrending()

    # Calculate Minkowski's standard for previously set 'power' and 'sets' values (lab3)
    # lab.choose_sets_lab3(power, sets, choice)

    # Calculate distance for given moving window width (lab 4 task 1)
    # power = 2
    # width = 5
    # lab.lab4_zad1(width, power)

    # Calculate distance for specified 3 windows widths (lab 4 task 2)`
    # lab.lab4_zad1(1, 2)
    # lab.lab4_zad2(power)
    # lab.lab4_zad3()

    # lab 5
    ran = [[100, 200], [400, 650], [1000, 1200], [1600, 1660], [2000, 2200]]
    r = lab.lab5_zad2(ran)
    lab.lab5_zad3(r, ran)

