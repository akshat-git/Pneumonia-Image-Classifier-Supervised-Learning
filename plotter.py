import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def extract_train_acc_values(input_string,name):
    # Use a regex to find all values following "Train Acc:"
    pattern = fr"{re.escape(name)}\s*([\d.]+)"
    train_acc_values = re.findall(pattern, input_string)
    return train_acc_values

# Example usage
input_string = """
Valid loss decreased inf --> 0.5197163224220276
Epoch 1: Train Loss: 0.24936834962730028, Train Acc: 0.8941717743873596
Epoch 1: Valid Loss: 0.5197163224220276, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.5197163224220276 --> 0.49762681126594543
Epoch 2: Train Loss: 0.1439747953518905, Train Acc: 0.9394171833992004
Epoch 2: Valid Loss: 0.49762681126594543, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.49762681126594543 --> 0.45094358921051025
Epoch 3: Train Loss: 0.1367689968815982, Train Acc: 0.9442101120948792
Epoch 3: Valid Loss: 0.45094358921051025, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.45094358921051025 --> 0.3360065817832947
Epoch 4: Train Loss: 0.11974923685835846, Train Acc: 0.952262282371521
Epoch 4: Valid Loss: 0.3360065817832947, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 5: Train Loss: 0.10684656233675321, Train Acc: 0.9587806463241577
Epoch 5: Valid Loss: 0.39489588141441345, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 6: Train Loss: 0.1074676966388207, Train Acc: 0.9593558311462402
Epoch 6: Valid Loss: 0.5097837448120117, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 7: Train Loss: 0.10669142837876146, Train Acc: 0.9603143930435181
Epoch 7: Valid Loss: 0.3960518538951874, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.3360065817832947 --> 0.31325677037239075
Epoch 8: Train Loss: 0.09639582954999916, Train Acc: 0.9643405079841614
Epoch 8: Valid Loss: 0.31325677037239075, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 9: Train Loss: 0.09771008381442926, Train Acc: 0.9622315764427185
Epoch 9: Valid Loss: 0.6826381683349609, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 10: Train Loss: 0.09772616885338219, Train Acc: 0.9593558311462402
Epoch 10: Valid Loss: 0.5341136455535889, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 11: Train Loss: 0.098471942930868, Train Acc: 0.9631901979446411
Epoch 11: Valid Loss: 0.3327411711215973, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 12: Train Loss: 0.08882000520818244, Train Acc: 0.9654908180236816
Epoch 12: Valid Loss: 0.8646448254585266, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 13: Train Loss: 0.08927014836737246, Train Acc: 0.9675996899604797
Epoch 13: Valid Loss: 0.3235965371131897, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 14: Train Loss: 0.09585805139500803, Train Acc: 0.9635736346244812
Epoch 14: Valid Loss: 0.5023179650306702, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.31325677037239075 --> 0.281697154045105
Epoch 15: Train Loss: 0.09662730539606118, Train Acc: 0.9631901979446411
Epoch 15: Valid Loss: 0.281697154045105, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 16: Train Loss: 0.0820648004243756, Train Acc: 0.9718174934387207
Epoch 16: Valid Loss: 0.37300607562065125, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 17: Train Loss: 0.08834939877084097, Train Acc: 0.9660659432411194
Epoch 17: Valid Loss: 0.7262156009674072, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 18: Train Loss: 0.0836346808322716, Train Acc: 0.9660659432411194
Epoch 18: Valid Loss: 1.0323506593704224, Valid Acc: 0.5625
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 19: Train Loss: 0.07699106655874935, Train Acc: 0.9712423086166382
Epoch 19: Valid Loss: 0.29756051301956177, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 20: Train Loss: 0.08362783103619925, Train Acc: 0.9660659432411194
Epoch 20: Valid Loss: 0.4331577718257904, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 21: Train Loss: 0.08951663156789927, Train Acc: 0.9635736346244812
Epoch 21: Valid Loss: 0.4587036371231079, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 22: Train Loss: 0.08131971210027056, Train Acc: 0.9685583114624023
Epoch 22: Valid Loss: 0.3176296055316925, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 23: Train Loss: 0.08264367055559269, Train Acc: 0.9677914381027222
Epoch 23: Valid Loss: 0.503498911857605, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 24: Train Loss: 0.07734719482307648, Train Acc: 0.970475435256958
Epoch 24: Valid Loss: 0.6937887668609619, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 25: Train Loss: 0.075707992743965, Train Acc: 0.9712423086166382
Epoch 25: Valid Loss: 0.32564038038253784, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 26: Train Loss: 0.08290838151834505, Train Acc: 0.9697085618972778
Epoch 26: Valid Loss: 0.4501185715198517, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 27: Train Loss: 0.07266786974959041, Train Acc: 0.9714340567588806
Epoch 27: Valid Loss: 0.5473800301551819, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 28: Train Loss: 0.07060718533413005, Train Acc: 0.9745015501976013
Epoch 28: Valid Loss: 0.37030866742134094, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 29: Train Loss: 0.07382087585815526, Train Acc: 0.9720091819763184
Epoch 29: Valid Loss: 0.29225340485572815, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 30: Train Loss: 0.07257153220086193, Train Acc: 0.9722009301185608
Epoch 30: Valid Loss: 0.3393498361110687, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 31: Train Loss: 0.0710240109371285, Train Acc: 0.9735429286956787
Epoch 31: Valid Loss: 0.3461180031299591, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.281697154045105 --> 0.23206496238708496
Epoch 32: Train Loss: 0.07737538187242858, Train Acc: 0.9708588719367981
Epoch 32: Valid Loss: 0.23206496238708496, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 33: Train Loss: 0.07567689742605871, Train Acc: 0.9710506200790405
Epoch 33: Valid Loss: 0.42893075942993164, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 34: Train Loss: 0.06801633428774369, Train Acc: 0.9754601120948792
Epoch 34: Valid Loss: 0.282164067029953, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 35: Train Loss: 0.07537742182239247, Train Acc: 0.9691334366798401
Epoch 35: Valid Loss: 0.6608191132545471, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 36: Train Loss: 0.07699683673776056, Train Acc: 0.9708588719367981
Epoch 36: Valid Loss: 0.836469829082489, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 37: Train Loss: 0.06587526982400993, Train Acc: 0.9743098020553589
Epoch 37: Valid Loss: 0.5801079273223877, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 38: Train Loss: 0.060631492479948576, Train Acc: 0.9771856069564819
Epoch 38: Valid Loss: 0.5507760047912598, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 39: Train Loss: 0.07238720920764659, Train Acc: 0.974693238735199
Epoch 39: Valid Loss: 0.4581667482852936, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 40: Train Loss: 0.07346077939492587, Train Acc: 0.9723926186561584
Epoch 40: Valid Loss: 0.3505116105079651, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 41: Train Loss: 0.07207762284427195, Train Acc: 0.9712423086166382
Epoch 41: Valid Loss: 0.5234410166740417, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 42: Train Loss: 0.06492309182013836, Train Acc: 0.9768021702766418
Epoch 42: Valid Loss: 0.3551842272281647, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 43: Train Loss: 0.07054779925965704, Train Acc: 0.974693238735199
Epoch 43: Valid Loss: 0.4656383991241455, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.23206496238708496 --> 0.22664953768253326
Epoch 44: Train Loss: 0.06730811050832357, Train Acc: 0.9748849868774414
Epoch 44: Valid Loss: 0.22664953768253326, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 45: Train Loss: 0.06360094362159761, Train Acc: 0.9781441688537598
Epoch 45: Valid Loss: 0.26224836707115173, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 46: Train Loss: 0.060395785899596446, Train Acc: 0.9794861674308777
Epoch 46: Valid Loss: 0.3904975652694702, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 47: Train Loss: 0.06795423952923328, Train Acc: 0.9723926186561584
Epoch 47: Valid Loss: 0.2876078188419342, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 48: Train Loss: 0.07424046527074744, Train Acc: 0.9708588719367981
Epoch 48: Valid Loss: 0.33288687467575073, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 49: Train Loss: 0.06152030054765162, Train Acc: 0.9756518602371216
Epoch 49: Valid Loss: 0.2469431310892105, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 50: Train Loss: 0.06544471937828081, Train Acc: 0.9764187335968018
Epoch 50: Valid Loss: 0.47934582829475403, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 51: Train Loss: 0.06992938765387145, Train Acc: 0.9722009301185608
Epoch 51: Valid Loss: 0.40081852674484253, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 52: Train Loss: 0.0641013957373298, Train Acc: 0.9758435487747192
Epoch 52: Valid Loss: 0.22799454629421234, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 53: Train Loss: 0.06106858397173364, Train Acc: 0.9785276055335999
Epoch 53: Valid Loss: 0.2928912043571472, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.22664953768253326 --> 0.2165343314409256
Epoch 54: Train Loss: 0.06929039806584639, Train Acc: 0.972967803478241
Epoch 54: Valid Loss: 0.2165343314409256, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.2165343314409256 --> 0.1788214147090912
Epoch 55: Train Loss: 0.06102017940581273, Train Acc: 0.9769938588142395
Epoch 55: Valid Loss: 0.1788214147090912, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 56: Train Loss: 0.06413115251900776, Train Acc: 0.9750766754150391
Epoch 56: Valid Loss: 0.28687334060668945, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 57: Train Loss: 0.06304737063498127, Train Acc: 0.9798696041107178
Epoch 57: Valid Loss: 0.22222943603992462, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 58: Train Loss: 0.06269021314810576, Train Acc: 0.9766104221343994
Epoch 58: Valid Loss: 0.2108278125524521, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 59: Train Loss: 0.0593639083817834, Train Acc: 0.9783359169960022
Epoch 59: Valid Loss: 0.19255167245864868, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Valid loss decreased 0.1788214147090912 --> 0.17093342542648315
Epoch 60: Train Loss: 0.06433728265135401, Train Acc: 0.9760352969169617
Epoch 60: Valid Loss: 0.17093342542648315, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 61: Train Loss: 0.05938651902352758, Train Acc: 0.977569043636322
Epoch 61: Valid Loss: 0.4961424767971039, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 62: Train Loss: 0.059768355993672376, Train Acc: 0.9781441688537598
Epoch 62: Valid Loss: 0.528353750705719, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 63: Train Loss: 0.05433550427466569, Train Acc: 0.9789110422134399
Epoch 63: Valid Loss: 0.33553066849708557, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 64: Train Loss: 0.05717023983886115, Train Acc: 0.9773772954940796
Epoch 64: Valid Loss: 0.3541363775730133, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 65: Train Loss: 0.0649547196069516, Train Acc: 0.9748849868774414
Epoch 65: Valid Loss: 0.5077093243598938, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 66: Train Loss: 0.05914123728157141, Train Acc: 0.9769938588142395
Epoch 66: Valid Loss: 0.19953876733779907, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 67: Train Loss: 0.06520051882099474, Train Acc: 0.9745015501976013
Epoch 67: Valid Loss: 0.252081036567688, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 68: Train Loss: 0.054434950193415245, Train Acc: 0.9789110422134399
Epoch 68: Valid Loss: 0.18399083614349365, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 69: Train Loss: 0.05740673392221897, Train Acc: 0.9781441688537598
Epoch 69: Valid Loss: 0.4424370229244232, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 70: Train Loss: 0.05573356355745074, Train Acc: 0.977569043636322
Epoch 70: Valid Loss: 0.3565495014190674, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 71: Train Loss: 0.059737944283431245, Train Acc: 0.9766104221343994
Epoch 71: Valid Loss: 0.22734016180038452, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 72: Train Loss: 0.059663880943085716, Train Acc: 0.9769938588142395
Epoch 72: Valid Loss: 0.6728389859199524, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 73: Train Loss: 0.05673871084131219, Train Acc: 0.9798696041107178
Epoch 73: Valid Loss: 0.45175856351852417, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 74: Train Loss: 0.058576732186316316, Train Acc: 0.9768021702766418
Epoch 74: Valid Loss: 0.521051824092865, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 75: Train Loss: 0.0566986676389139, Train Acc: 0.9783359169960022
Epoch 75: Valid Loss: 0.265586256980896, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 76: Train Loss: 0.058480518644188524, Train Acc: 0.977569043636322
Epoch 76: Valid Loss: 0.6524500846862793, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 77: Train Loss: 0.05754969938130738, Train Acc: 0.9764187335968018
Epoch 77: Valid Loss: 0.7334539890289307, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 78: Train Loss: 0.05776948231206157, Train Acc: 0.9771856069564819
Epoch 78: Valid Loss: 0.18629738688468933, Valid Acc: 1.0
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 79: Train Loss: 0.05310116603335243, Train Acc: 0.9796779155731201
Epoch 79: Valid Loss: 0.4684443473815918, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 80: Train Loss: 0.05710178757422028, Train Acc: 0.9779524803161621
Epoch 80: Valid Loss: 0.5495858788490295, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 81: Train Loss: 0.05488276550718769, Train Acc: 0.9787193536758423
Epoch 81: Valid Loss: 0.2724654972553253, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 82: Train Loss: 0.05279416994088504, Train Acc: 0.9804447889328003
Epoch 82: Valid Loss: 0.2683897614479065, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 83: Train Loss: 0.049955427488351405, Train Acc: 0.9829370975494385
Epoch 83: Valid Loss: 0.37580034136772156, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 84: Train Loss: 0.0563428179217014, Train Acc: 0.9794861674308777
Epoch 84: Valid Loss: 0.6056531667709351, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 85: Train Loss: 0.05436756354487041, Train Acc: 0.9794861674308777
Epoch 85: Valid Loss: 0.20723792910575867, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 86: Train Loss: 0.057962437589828976, Train Acc: 0.977569043636322
Epoch 86: Valid Loss: 0.7630881071090698, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 87: Train Loss: 0.052615467882821405, Train Acc: 0.9808282256126404
Epoch 87: Valid Loss: 0.3015037775039673, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 88: Train Loss: 0.05616526313390449, Train Acc: 0.9777607321739197
Epoch 88: Valid Loss: 0.6687844395637512, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 89: Train Loss: 0.05092633636924099, Train Acc: 0.981019914150238
Epoch 89: Valid Loss: 0.19305558502674103, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 90: Train Loss: 0.0551547835295055, Train Acc: 0.97929447889328
Epoch 90: Valid Loss: 0.32819873094558716, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 91: Train Loss: 0.04793191726706591, Train Acc: 0.9815950989723206
Epoch 91: Valid Loss: 0.2974032163619995, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 92: Train Loss: 0.0499421066237275, Train Acc: 0.9804447889328003
Epoch 92: Valid Loss: 0.558759331703186, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 93: Train Loss: 0.05502919464840259, Train Acc: 0.9783359169960022
Epoch 93: Valid Loss: 0.18729805946350098, Valid Acc: 0.9375
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 94: Train Loss: 0.0505820228569131, Train Acc: 0.9808282256126404
Epoch 94: Valid Loss: 0.24231204390525818, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 95: Train Loss: 0.053428442875660585, Train Acc: 0.9785276055335999
Epoch 95: Valid Loss: 0.5670610666275024, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 96: Train Loss: 0.05312386536587296, Train Acc: 0.9812116622924805
Epoch 96: Valid Loss: 0.4151805639266968, Valid Acc: 0.75
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 97: Train Loss: 0.05248562217016422, Train Acc: 0.9819785356521606
Epoch 97: Valid Loss: 0.2721197307109833, Valid Acc: 0.8125
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 98: Train Loss: 0.055054233182079526, Train Acc: 0.9802530407905579
Epoch 98: Valid Loss: 0.2381070852279663, Valid Acc: 0.875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 99: Train Loss: 0.049003725868229826, Train Acc: 0.9829370975494385
Epoch 99: Valid Loss: 0.4499817192554474, Valid Acc: 0.6875
  0%|          | 0/326 [00:00<?, ?it/s]
  0%|          | 0/1 [00:00<?, ?it/s]
Epoch 100: Train Loss: 0.054238296401874955, Train Acc: 0.97929447889328
Epoch 100: Valid Loss: 0.6168990731239319, Valid Acc: 0.6875
"""

#train_acc = extract_train_acc_values(input_string,'Train Acc:')
#train_loss = extract_train_acc_values(input_string,'Train Loss:')

def plot_acc_loss(train_acc,train_loss):
    # Generate x-coordinates (index + 1)
    x_coords = list(range(1, len(train_acc) + 1))
    round_train_acc = [round(float(acc),4) for acc in train_acc]
    round_train_loss = [round(float(loss),4) for loss in train_loss]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, round_train_acc, marker='o', linestyle='-', color='b', label='Values')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title(f'Plot of Train Accuracy over {len(train_acc)} Epochs')

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.legend()

    # Show the plot
    plt.show()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, round_train_loss, marker='o', linestyle='-', color='b', label='Values')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(f'Plot of Train Loss over {len(train_acc)} Epochs')

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.legend()

    # Show the plot
    plt.show()
