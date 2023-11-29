from data_preprocess import *
from transformer import *
def test(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    predicts = []
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob  = projected.squeeze(0).max(dim = -1, keepdim = False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
        predicts.append(next_symbol)
    return predicts

enc_inputs, dec_inputs, dec_outputs = make_data()
loader = Data.DataLoader(MyDataset(enc_inputs, dec_inputs, dec_outputs), 2, True)
enc_inputs, _, _ = next(iter(loader))
model = torch.load('transformer.pth')
predict_dec_input = test(model, enc_inputs[0].view(1,-1).cuda(), start_symbol=tgt_vocab["S"])
print([src_idx2word[int(i)] for i in enc_inputs[0]], "->", [tgt_idx2word[n] for n in predict_dec_input])
