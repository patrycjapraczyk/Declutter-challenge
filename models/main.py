from models.model_exec import ModelExec


def main():
    exec = ModelExec(comment_vectoriser='BOW')
    #exec.kfold_split(10, 0.5454545454545454, 0.45454545454545453, exec.data)
    exec.test_python_data('ADASYN')

if __name__ == "__main__":
    main()