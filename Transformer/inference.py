from utils import subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = src.new_zeros(1,1).fill_(start_symbol)

    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           ys,
                           subsequent_mask(ys.size(1)).type_as(src_mask))
        prob = model.generator(out[:,-1])
        next_word = prob.argmax(dim=-1)
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)

    return ys
