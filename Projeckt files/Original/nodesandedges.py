import sumolib

def netzwerk_info(net_file1,net_file2):
    net1,net2 = sumolib.net.readNet(net_file1), sumolib.net.readNet(net_file2)
    knoten1,knoten2 = net1.getNodes(), net2.getNodes()
    kanten1, kanten2 = net1.getEdges(), net2.getEdges()

    knoten_anzahl1, knoten_anzahl2 = len(knoten1),len(knoten2)
    kanten_anzahl1,kanten_anzahl2 = len(kanten1), len(kanten2)

    original_nodes_difference = knoten_anzahl1 - knoten_anzahl2
    cleaned_nodes_difference = knoten_anzahl2 - knoten_anzahl1

    if knoten_anzahl1 > knoten_anzahl2:
        print(f'Original hat {original_nodes_difference} Knoten mehr als das neue')
    elif knoten_anzahl1 < knoten_anzahl2:
        print(f'Das Neue hat {cleaned_nodes_difference} Knoten mehr als das originale')
    else:
        print('Anzahl von Knoten fuer beides ist gleich')

    original_edges_difference = kanten_anzahl1 - kanten_anzahl2
    cleaned_edges_difference = kanten_anzahl2 - kanten_anzahl1
    
    if kanten_anzahl1 > kanten_anzahl2:
        print(f'Original hat {original_edges_difference} Kanten mehr als das neue')
    elif kanten_anzahl1 < kanten_anzahl2:
        print(f'Das Neue hat {cleaned_edges_difference} Kanten mehr als das originale')
    else:
        print('Anzahl von Kanten fuer beides ist gleich')
    

original_net = 'map_m.net.xml'
cleaned_net = 'berlin_original.net.xml'
netzwerk_info(original_net,cleaned_net)

