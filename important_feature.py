# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:32:18 2025

@author: THINKPAD
"""
from collections import Counter

a = [405.0, 524.0, 403.0, 689.0, 551.0, 802.0, 527.0, 400.0, 530.0, 549.0, 900.0, 589.0, 692.0, 473.0, 519.0, 538.0, 576.0, 929.0, 532.0]

b = [524.0, 557.0, 576.0, 689.0, 403.0, 800.0, 554.0, 486.0, 802.0, 522.0, 405.0, 551.0, 470.0, 516.0, 962.0, 408.0, 627.0, 527.0, 481.0, 454.0]

c = [524.0, 405.0, 689.0, 557.0, 551.0, 800.0, 576.0, 527.0, 802.0, 554.0, 589.0, 522.0, 532.0, 997.0, 478.0, 538.0, 686.0, 927.0, 470.0, 603.0]

d = [524.0, 689.0, 551.0, 527.0, 557.0, 800.0, 405.0, 576.0, 802.0, 540.0, 530.0, 516.0, 478.0, 403.0, 486.0, 565.0, 476.0, 554.0, 459.0, 589.0]

e = [997.0, 400.0, 994.0, 411.0, 432.0, 408.0, 451.0, 983.0, 924.0, 403.0, 414.0, 978.0, 929.0, 505.0, 927.0, 459.0, 405.0, 989.0, 991.0, 500.0]

f = [495.0, 592.0, 503.0, 813.0, 500.0, 705.0, 816.0, 468.0, 454.0, 492.0, 465.0, 508.0, 497.0, 543.0, 546.0, 538.0, 513.0, 732.0, 975.0, 505.0]

g = [524.0, 689.0, 495.0, 813.0, 405.0, 503.0, 457.0, 592.0, 705.0, 802.0, 816.0, 513.0, 527.0, 468.0, 486.0, 702.0, 492.0, 546.0, 557.0, 500.0, 454.0, 497.0, 451.0, 800.0, 538.0, 576.0]

numbers= a + b + c + d + e + f + g

count = Counter(numbers)
filtered_numbers = [item[0] for item in Counter(numbers).items() if item[1] >= 2]
filtered_numbers_sorted = sorted(filtered_numbers, key=lambda x: Counter(numbers)[x], reverse=True)
print(count)
print(filtered_numbers_sorted)


h = [405.0, 419.0, 524.0, 797.0, 905.0, 559.0, 802.0, 408.0, 608.0, 505.0, 962.0, 816.0, 900.0, 689.0, 951.0, 740.0, 530.0, 843.0, 519.0]

i = [667.0, 524.0, 549.0, 554.0, 551.0, 800.0, 689.0, 816.0, 975.0, 754.0, 557.0, 576.0, 924.0, 962.0, 594.0, 835.0, 751.0, 756.0, 719.0, 657.0]

j = [524.0, 800.0, 816.0, 665.0, 994.0, 813.0, 441.0, 883.0, 746.0, 862.0, 864.0, 532.0, 468.0, 603.0, 978.0, 465.0, 735.0, 621.0, 576.0, 567.0]

k = [581.0, 875.0, 459.0, 781.0, 638.0, 565.0, 792.0, 530.0, 778.0, 532.0, 762.0, 651.0, 527.0, 819.0, 816.0, 592.0, 832.0, 802.0, 951.0, 929.0]

l = [495.0, 603.0, 810.0, 754.0, 981.0, 697.0, 597.0, 802.0, 713.0, 716.0, 524.0, 551.0, 592.0, 449.0, 446.0, 522.0, 651.0, 694.0, 465.0, 594.0]

m = [581.0, 557.0, 524.0, 551.0, 527.0, 816.0, 486.0, 405.0, 665.0, 495.0, 746.0, 905.0, 918.0, 689.0, 492.0, 810.0, 802.0, 624.0, 792.0, 800.0, 611.0, 927.0, 805.0, 548.0, 581.0, 732.0, 516.0]

numbers2= h + i + j + k + l + m

count_2 = Counter(numbers2)
filtered_numbers_2 = [item[0] for item in Counter(numbers2).items() if item[1] >= 2]
filtered_numbers_sorted_2 = sorted(filtered_numbers_2, key=lambda x: Counter(numbers2)[x], reverse=True)
print(count_2)
print(filtered_numbers_sorted_2)


numbers3 = numbers + numbers2
count_3 = Counter(numbers3)
filtered_numbers_3 = [item[0] for item in Counter(numbers3).items() if item[1] >= 2]
filtered_numbers_sorted_3 = sorted(filtered_numbers_3, key=lambda x: Counter(numbers3)[x], reverse=True)
print(count_3)
print(filtered_numbers_sorted_3)