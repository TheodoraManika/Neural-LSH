# Neural-LSH Build - Οδηγίες Χρήσης

## Περιγραφή
Το `nlsh_build.py` είναι υπεύθυνο για την κατασκευή του Neural-LSH ευρετηρίου.

## Βήματα που Εκτελεί

1. **Φόρτωση Δεδομένων**: Ανάγνωση του dataset
2. **Κατασκευή k-NN Γράφου**: Εύρεση k-πλησιέστερων γειτόνων, συμμετρικοποίηση, απόδοση βαρών
3. **Διαμέριση Γράφου**: Χρήση KaHIP για balanced partitioning σε K buckets
4. **Εκπαίδευση MLP**: Εκπαίδευση 3-στρωμάτων νευρωνικού δικτύου
5. **Αποθήκευση**: Αποθήκευση μοντέλου και ευρετηρίου

## Χρήση

### Βασική Χρήση
```bash
python nlsh_build.py --dataset path/to/dataset.npy --k 10 --K 100
```

### Παράμετροι

- `--dataset`: **[ΑΠΑΙΤΕΙΤΑΙ]** Διαδρομή προς το αρχείο δεδομένων
- `--k`: Αριθμός πλησιέστερων γειτόνων (default: 10)
- `--K`: Αριθμός buckets/διαμερίσεων (default: 100)
- `--hidden_dim1`: Διάσταση 1ου κρυφού επιπέδου (default: 128)
- `--hidden_dim2`: Διάσταση 2ου κρυφού επιπέδου (default: 64)
- `--epochs`: Αριθμός εποχών εκπαίδευσης (default: 50)
- `--batch_size`: Μέγεθος batch (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--model_path`: Διαδρομή αποθήκευσης μοντέλου (default: model/nlsh_model.pth)
- `--index_path`: Διαδρομή αποθήκευσης ευρετηρίου (default: index/nlsh_index.pkl)

### Παραδείγματα

```bash
# Βασική εκτέλεση με default παραμέτρους
python nlsh_build.py --dataset dataset/sift_base.npy

# Προσαρμοσμένες παράμετροι
python nlsh_build.py \
    --dataset dataset/mnist.npy \
    --k 20 \
    --K 200 \
    --hidden_dim1 256 \
    --hidden_dim2 128 \
    --epochs 100 \
    --batch_size 512 \
    --lr 0.0005

# Καθορισμός διαδρομών αποθήκευσης
python nlsh_build.py \
    --dataset dataset/data.npy \
    --model_path model/my_model.pth \
    --index_path index/my_index.pkl
```

## Έξοδος

Το πρόγραμμα δημιουργεί:

1. **Μοντέλο** (`.pth`): Εκπαιδευμένο PyTorch MLP μοντέλο
2. **Ευρετήριο** (`.pkl`): Οργανωμένα δεδομένα ανά bucket

## Δομή Αρχείων

```
nlsh/
├── graph/
│   ├── __init__.py
│   └── knn_graph.py          # Κατασκευή k-NN γράφου
└── model_index_init/
    ├── __init__.py
    ├── mlp_model.py           # Ορισμός MLP
    └── index_builder.py       # Κατασκευή ευρετηρίου
```

## Σημειώσεις

- Το dataparser για binary files (.fvecs, .ivecs, .bvecs) θα υλοποιηθεί ξεχωριστά
- Προς το παρόν υποστηρίζονται `.npy` αρχεία
- Το πρόγραμμα χρησιμοποιεί GPU αν είναι διαθέσιμο, αλλιώς CPU
