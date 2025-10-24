Celem projektu było opracowanie rozwiązania umożliwiającego klasyfikację obrazów rentgenowskich płuc w kierunku zapalenia płuc z wykorzystaniem metod uczenia głębokiego.
W pracy zastosowano konwolucyjne sieci neuronowe (CNN) zaimplementowane w środowisku Jupyter Notebook z użyciem biblioteki PyTorch.
Przeprowadzono proces wstępnego przetwarzania danych pochodzących z różnych źródeł oraz trening i ewaluację modeli głębokiego uczenia.
Kluczowym elementem projektu było wykorzystanie map aktywacji (Grad-CAM) do wizualizacji obszarów obrazu istotnych dla decyzji klasyfikacyjnych modelu, co zwiększyło interpretowalność wyników.
Opracowane rozwiązanie pozwala na analizę skuteczności sieci CNN w diagnostyce zapalenia płuc oraz prezentuje potencjał sztucznej inteligencji w analizie obrazów medycznych.

Zestaw danych pochodzi z publicznego zbioru Chest X-Ray Images (Pneumonia) dostępnego na Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Jak uruchomić projekt?

Sklonuj repozytorium:
~ git clone https://github.com/werkazana/Projekt_inzynierski.git
~ cd Projekt_inzynierski

Pobierz dane z Kaggle i wypakuj je do folderu:
~ chest_xray/

Zainstaluj wymagane biblioteki:
~ pip install -r requirements.txt

Uruchom notebook:
~ pneumoniavg.ipynb
