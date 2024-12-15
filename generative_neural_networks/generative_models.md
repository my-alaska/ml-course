[//]: # (compile with command `pandoc generative_models.md -o generative_models.pdf`)

![](data/report/Znak_graficzny_AGH.svg.png)

# Modele Generatywne - Sprawozdanie

Piotr Ludynia - Informatyka Data Science

Uczenie Maszynowe

---

\newpage

# Zadanie 1

### Zadanie 1.1 

Dlaczego sigmoid jest odpowiednią funkcją aktywacji w ostatniej warstwie dekodera w tym przypadku?

Odp.: Chcemy przewidywać wartości z przedziału `(0,1)`. Sigmoid mapuje cały przedział liczb rzeczywistych właśnie na te wartości.

---

### Zadanie 1.2

Skompiluj model. 
W tym celu najpierw zdefiniuj loss dla modelu. 
W przypadku autoenkodera jest to funkcja działająca na wejściach do enkodera oraz wyjściach z dekodera. 
Do wyboru są różne funkcje! 
Patrząc na reprezentację danych (wróć do funkcji definiującej preprocessing), 
wybierz odpowiednią. Uzasadnij swój wybór.

Odp.: Problemem jest regresja, ale w ograniczonym przedziale wartości. 
Pierwszym pomysłem na funkcję kosztu było użycie binarnej entropii krzyżowej i modelowanie zadania jako swego rodzaju klasyfikacja. 
Wartości są jednak ciągłe, więc użyto MSE.

Wyniki treningu klasycznego autoenkodera. Przedział wrysowanej siatki ustawiono na [-4;4] i [-4;4]. Różnymi kolorami zaznaczono poszczególne cyfry.

| ![fig1](data/report/latent_space_AE.png) |     ![fig2](data/report/grid_ae.png)      |
|:----------------------------------------:|:-----------------------------------------:|
|       *Fig. 1 - Latent Space - AE*       |           *Fig. 2 - Grid - AE*            |

---

\newpage

### Zadanie 1.3. 
Wybierz ze zbioru testowego dwa obrazy z różnymi liczbami. 
Dobierz takie liczby, dla których spodziewasz się, że odkodowanie średniej z ich zenkodowanych reprezentacji będzie miało sens. 
Wybierz dwie takie pary.  

Dla każdej z par:  
Wyświetl wybrane liczby.

* Wybrano pary (9,7) oraz (1,8)
    
| ![fig3](data/report/2_pairs_ae.png) |
|:-----------------------------------:|
|   *Fig. 3 - two pairs of numbers*   | 

Użyj enkodera do uzyskania 2-wymiarowych reprezentacji każdej liczby.

Wylicz średnią z tych reprezentacji.  

Użyj dekodera na uzyskanej średniej.  

Wyświetl wynik.

* By lepiej zwizualizować przejście, wybrano 3 liczby pomiędzy zakodowanymi wartościami. Pierwsza i ostatnia wartość w rzędzie to zdekodowane osadzenia oryginalnych obrazów. 3 obrazy pośrodku pochodzą z odkodowań punktów równo ułożonych na prostej między nimi. W szczególności środkowe obrazy są średnią z położeń 2 wektorów.

| ![fig4](data/report/phase_change_ae.png) |
|:----------------------------------------:|
| *Fig. 4 - images between the pairs - AE* | 

Skomentuj wynik - czy przypomina jakąś liczbę? Czy takiego wyniku się spodziewałaś/eś?

* Odtworzenie osadzenia cyfry 8 bardziej przypomina cyfrę 5. Środkowe obrazy faktycznie wyglądają jak nieco zmieszane cyfry odpowiednio 9 z 7 oraz 1 z 8 - środkowy obraz przypomina bardzo spłaszczoną cyfrę 8. Wyniki nie są jednak zadowalające

---

\newpage

# Zadanie 2

### Zadanie 2.1

Dlaczego implementacje CVAE nie stosuje żadnej aktywacji w ostatniej warstwie enkodera? Czy jakaś funkcja by się tutaj nadawała?

Chcemy, by enkodowane wektory były elementami przestrzeni n-wymiarowej rozłożonymi wokół środka układu współrzędnych. Stosowanie aktywacji sprawiłoby, że n-wymiarowa przestrzeń zostałaby zmarszczona lub przycięta, co utrudniłoby próbkowanie.

---

Wyniki treningu dla autoenkodera wariacyjnego

| ![fig5](data/report/training_result_vae.png) |
|:--------------------------------------------:|
|       *Fig. 5 - Training result - VAE*       | 

---

\newpage

### Zadanie 2.2

Skomentuj wynik uzyskany przy użyciu funkcji plot_latent_images. Zwróć uwagę na jakość/sensowność rysowanych liczb. Porównaj wykres do analogicznego wykresu dla modelu AE. Zamieść w raporcie wykresy.  

* Więcej obrazków wykazuje rozmycie, jednak jest ono dużo mniejsze niż na zupełnie rozmytych obrazach z pierwszej metody. (Wykres poniżej)
  
---

### Zadanie 2.3

Porównaj wyniki funkcji _plot_latent_space_ dla AE oraz VAE. Zwróć uwagę na "gęstość" punktów oraz zakres wartości. Zamieść w raporcie wykresy.  

* Punkty pokrywają przestrzeń bardziej równomiernie dla VAE niż AE. Nie tworzą aż tak ciasnych skupisk a przestrzenie między takimi klastrami są bardziej zapełnione. 


| ![fig6](data/report/latent_space_VAE.png) |     ![fig7](data/report/grid_Vae.png)     |
|:-----------------------------------------:|:-----------------------------------------:|
|       *Fig. 6 - Latent Space - VAE*       |           *Fig. 7 - Grid - VAE*           |


---

\newpage

### Zadanie 2.4

Dla tych samych par obrazów, na których pracowałaś/eś w ostatnim zadaniu dot. AE, przygotuj reprezentacje ukryte z pomocą wytrenowanego VAE i odkoduj średnie z reprezentacji. Skomentuj wyniki, porównaj z wynikami z AE.

* Generacja danych nie przyniosła dużo lepszych rezultatów niż wcześniej. Widać to szczególnie dobrze dla wartości zkwantyzowanych do 0 i 1. Odtworzenie osadzeń obrazów wejściowych - cyfr 9 i 8 nie przypominają zupełnie oryginałów. 

| ![fig8](data/report/2_pairs_quantized_vae.png) | ![fig9](data/report/phase_change_vae.png) |
|:-----------------------------------------:|:----------------------------------------------:|
|        *Fig. 8 - Pairs of images - VAE*        |      *Fig. 9 - Images between pairs*      |


---

\newpage

### Zadanie 2.5

Wróć do funkcji *compute_loss*. Człony *logpz* oraz *logqz_x* związane są z obliczaniem KL-divergence pomiędzy *Q(z|X)* oraz *P(z)*. Zakładamy, że oba te rozkłady są gaussowskie, stąd możemy wykorzystać wzór na KL-divergence dla dwóch rozkładów gaussowskich. Znajdź ten wzór oraz przepisz funkcję *compute_loss* z jego wykorzystaniem. Zamieść w raporcie przygotowaną formułę. Wytrenuj model ponownie, porównaj wyniki z poprzednią implementacją *compute_loss*.

* Obliczony wzór na KL-divergence dwóch rozkładów normalnych. W pierwszej linii postać ogólna, a w drógiej przekształcona dla *P~N(0,1)*

| ![fig10](data/report/kld_formulas.png) |
|:--------------------------------------:|
|   *Fig. 10 - KL-divergence formula*    | 


* Niestety wyniki nie wydają się, być lepsze niż dla poprzedniej implementacji. Otrzymujemy bardzo podobne rozmycie.
    
| ![fig11](data/report/training_result_vae_special_loss.png) |
|:----------------------------------------------------------:|
|    *Fig. 11 - Training result - VAE with modified loss*    |

---

*Poprawę jakości modelu pominięto ze względu na czas konieczny do poświęcenia na dalsze części pracy*

---

\newpage

# Zadanie 3

### Zadanie 3.1

Sprawdź jakość modelu dla 3 różnych wartości latent_dim (trzeba dla każdej z nich osobno wytrenować model). 
Niech będą od siebie znacząco różne, np. 2, 25, 100. 
Przy większym latent_dim może być potrzebnych więcej epok.  

* Wszystkie trzy wyniki wyglądają dobrze. Do dalszej części eksperymentu wybrano wymiar 25. Dobre wyniki dla wymiaru 2 uzyskano po drobnych modyfikacjach architektury sieci.

|          ![fig12](data/report/cond_vae_2.png)          |      ![fig13](data/report/cond_vae_25.png)      |      ![fig14](data/report/cond_vae_100.png)      |
|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|
|     *Fig. 12 - Example results for latent space 2*     | *Fig. 13 - Example results for latent space 25* | *Fig. 14 - Example results for latent space 100* |

---

\newpage

### Zadanie 3.2

Wykonaj dla najlepszego modelu z punktu 3.1.:

Wybierz przykład ze zbioru testowego (obraz + etykieta).
    
| ![fig15](data/report/cond_representation.png)  |
|:----------------------------------------------:|
|           *Fig. 15 - Example image*            | 

Przepuść próbkę przez enkoder, uzyskaj reprezentację *z*.

Dla każdego z 9 możliwych wektorów [poprawna_etykieta, pos_x, pos_y] przepuść przez dekoder reprezentację _z_ wraz z informacją o etykiecie i położeniu. Wyświetl uzyskany obraz. Skomentuj wyniki.

|      ![fig16](data/report/cond_rep_all_positions.png)       |
|:-----------------------------------------------------------:|
|         *Fig. 16 - Representation on all positions*         | 

Udało się uzyskać obraz zgodny z oczekiwaniami. 
Sztucznie zakodowane pozycje dla osadzenia wygenerowanego z obrazu pozwoliły na umieszczenie poprawnych cyfr w zadanych pozycjach na obrazie. 
Można wyciągnąć wniosek, że wartości wektora osadzenia nie wpływają na pozycję na odkodowanym obrazie. 
A przynajmniej nie w tak wielkim stopniu, jak przekazywany warunek.

---

\newpage

### Zadanie 3.3

Powtórz zadanie 3.2, ale tym razem jako reprezentację *z* wykorzystaj wartości wylosowane z rozkładu normalnego oraz wybierz dowolną etykietę. 
Skomentuj wyniki - czy za każdym razem uzyskano oczekiwaną liczbę w oczekiwanym miejscu?

|               ![fig17](data/report/cond_rep_normal.png)               |
|:---------------------------------------------------------------------:|
|     *Fig. 17 - Representation on all positions - normal samples*      | 

Wszystkie liczby znajdują się w oczekiwanym miejscu. 
Wszystkie przypominają też zadaną w warunkach cyfrę 3. 
Tym razem wszystkie próbki z rozkładu normalnego w *latent_space* były różne. 
Widoczne są różne style, w których zapisano cyfrę 3.

---

\newpage

# Zadanie 4

Wygeneruj po jednym obrazie z każdą liczbą z pomocą generatora. 
Oceń jakość wyników. 
Jeśli jakość modelu pozostawia wiele do życzenia, spróbuj go poprawić, np. zwiększając liczbę epok bądź zmieniając definicję generatora/dyskryminatora.  

Pomimo długich prób wytrenowania GAN przez wiele epok i eksperymentów z rozmiarami batch-y, architekturą enkodera i dekodera, parametrami optymalizatora nie udało się uzyskać zadowalających wyników.

|  ![fig18](data/report/gan_results.png)   |
|:----------------------------------------:|
| *Fig. 18 - Final result of gan training* | 

Na obrazie 19 widać, że wygenerowane cyfry od 0 do 9 nie przypominają docelowych. Jedynie 3 i 5 lekko przypominają faktyczne kształty cyfr.

|      ![fig19](data/report/gan_examples.png)       |
|:-------------------------------------------------:|
| *Fig. 19 - Examples of digits generated with gan* | 