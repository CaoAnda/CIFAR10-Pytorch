import argparse  
  
def get_args(parser=argparse.ArgumentParser()):  
    parser.add_argument('--device', type=str, default='cuda:0')  
    parser.add_argument('--batch_size', type=int, default=128)  
    parser.add_argument('--epochs', type=int, default=40)  
    parser.add_argument('--lr', type=float, default=1e-2)  
    parser.add_argument('--seed', type=int, default=3407)  
    parser.add_argument('--shortcut_level', type=int, default=3)  

    parser.add_argument('--output',action='store_true',default=True,help="shows output")  
  
    opt = parser.parse_args()  
    if opt.output:  
        print(opt)
    return opt  
  
if __name__ == '__main__':  
    opt = get_args()
