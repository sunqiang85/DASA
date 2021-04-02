import json
def view_data():
    with open("tasks/R2R/data/R2R_train.json") as f:
        js = json.load(f)

    print(type(js))
    print(len(js))
    print(type(js[0]))
    print(js[0])
    num_ins = [len(c['instructions']) for c in js]
    print("instructions", sum(num_ins))

    num_nav = set([c['scan'] for c in js])
    print("scans", len(num_nav))

def view_cost():
    bert_layer = 9 # la
    vision_layer = 1
    cross_layer = 4 #
    h = 768
    v = 2048
    b =20
    l = 35


    embed_model = 900*h
    bert_layer_model = (h*768*3 + 768*3072*2)* bert_layer
    vision_layer_model = 2048*768
    cross_layer_model = (768*768*3*4)*cross_layer
    lstm_model = (768*768*4)


    embed_data = 80*768
    bert_layer_data = 80*(768+3072+768)*bert_layer # detach
    vision_layer_data = 80*2048*768 # detach
    cross_layer_data = (80*(768+768+3072+768)+36*(768+768+3072+768))*cross_layer # detach
    decoder_data = 80*768+768*3+768*3

    model = embed_model + bert_layer_model + vision_layer_model + cross_layer_model + lstm_model
    data = (embed_data + decoder_data) * b * l

    print("model:", model)
    print("data:", data)

def view_mcatt_cost():

    layer = 2
    seq_len = 20
    batch_size = 4

    embed_model = 900*300
    lstm_model = 768*768*4
    sga_model = (768*768*4)+(768*768*4)



    embed_data = 80*300
    lstm_data = 80*768
    vision_layer_data = (36*(768+768+3072+768))*layer
    cross_layer_data = (80*(768+768+3072+768)*768+36*(768+768+3072+768)*768)*layer # detach


    model = embed_model + + lstm_model + sga_model
    data = (embed_data+ lstm_data + vision_layer_data+cross_layer_data) * seq_len * batch_size
    print("model:", model)
    print("data:", data)


if __name__ == '__main__':
    view_mcatt_cost()
