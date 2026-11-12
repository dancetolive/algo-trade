# algotrade

Kripto (spot ve USDT teminatlı vadeli) ile tokenize altın üzerinde kural tabanlı
stratejileri **dürüstçe** test etmek ve sonra küçük bir hesapta çalıştırmak için
araç seti. Hedef hesap büyüklüğü 500 – 5.000 $.

## Önce beklenti

Garanti kazanç yok. Bireysel al-satçıların büyük çoğunluğu zarar ediyor ve geçmiş
testler her zaman gerçeğinden iyi görünür. Bu proje o yüzden üç şeyi ciddiye alır:
masraflar, ileriye bakma hataları ve "kaç denemeden biri tuttu" sorusu. Bir
stratejinin gerçek sınavı geçmiş test değil, ileriye dönük kâğıt işlemdir.

## Kurulum (Windows, PowerShell)

Gerekenler: Python 3.11 veya üstü, git, yaklaşık 10 GB boş disk (tüm veri için).

```powershell
cd C:\Users\Admin\Desktop\0-craft\19-trade
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
```

Son komut bütün testleri çalıştırır; hepsi geçmeli.

## Veriyi indirme

Veri Binance'in herkese açık arşivinden gelir (`data.binance.vision`), hesap ya da
anahtar gerekmez. İndirilen dosyalar `data\raw`, işlenmiş hali `data\parquet`
altına yazılır; ikisi de git'e girmez. Komutlar yarıda kalırsa yeniden
çalıştırılabilir, inmiş dosyalar atlanır.

```powershell
# 1. Sembol kataloğu (birkaç saniye)
.venv\Scripts\python -X utf8 scripts\fetch.py symbols

# 2. Günlük mumlar ve fonlama oranları: araştırmanın çoğu için yeterli (~110 MB, 10-20 dk)
.venv\Scripts\python -X utf8 scripts\fetch.py klines --market spot --interval 1d --symbols "@usdt"
.venv\Scripts\python -X utf8 scripts\fetch.py klines --market um --interval 1d --symbols "@usdt"
.venv\Scripts\python -X utf8 scripts\fetch.py funding --symbols "@usdt"

# 3. İsteğe bağlı: saatlik mumlar (~1,6 GB)
.venv\Scripts\python -X utf8 scripts\fetch.py klines --market spot --interval 1h --symbols "@usdt"
.venv\Scripts\python -X utf8 scripts\fetch.py klines --market um --interval 1h --symbols "@usdt"

# 4. İsteğe bağlı: büyük varlıkların 1 dakikalık mumları (~1,9 GB)
.venv\Scripts\python -X utf8 scripts\fetch.py klines --market um --interval 1m --since 2020-01 --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,PAXGUSDT,XAUUSDT
.venv\Scripts\python -X utf8 scripts\fetch.py klines --market spot --interval 1m --since 2020-01 --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,PAXGUSDT
```

İndirme sırasında veri temizlenir: aylık dosyalardaki eksik günler günlük
dosyalardan yamanır, işlem görmeyen (sıfır hacimli) dönemler atılır, saat başına
oturmayan mumlar hizalanır ve bir sembol yeniden listelenmiş ya da yeniden
adlandırılmışsa ayrı enstrümanlara bölünür (`LUNAUSDT` güncel olanı,
`LUNAUSDT__1` eskisini tutar).

## İlk çalıştırma

```powershell
.venv\Scripts\python research\baseline\trend_baseline.py
```

BTC ve ETH üzerinde en basit trend kuralını (kapanış, N günlük ortalamanın
üstündeyse tut) masraflar dahil test eder ve al-tut ile karşılaştırır. Bu bir
strateji önerisi değil, sonraki her fikrin geçmesi gereken çıtadır.

## Proje yapısı

| Yol | İçerik |
|---|---|
| `src/algotrade/backtest.py` | Mum bazlı portföy simülatörü. Zamanlama ve muhasebe kuralları dosyanın başında yazılı. |
| `src/algotrade/research.py` | Deneme çalıştırma ve kaydetme; borsa türlerine göre masraf ayarları. |
| `src/algotrade/stats.py` | Getiri istatistikleri, şans eseri olma olasılığı ölçüleri (PSR, deflated Sharpe, PBO). |
| `src/algotrade/validate.py` | İleriye bakma (look-ahead) testi. |
| `src/algotrade/signals.py`, `universe.py` | Sinyal yapı taşları; her tarihte o gün bilinen bilgiyle seçilen likit coin evreni. |
| `src/algotrade/data/` | Arşiv istemcisi (`vision.py`) ve veri yükleyiciler (`store.py`). |
| `scripts/fetch.py` | Veri indirme ve yeniden üretme. |
| `research/` | Araştırma kolları: referans strateji, 1 dakikalık yapı çalışması. |
| `reports/trials/` | Denenen her ayarın kaydı. |
| `tests/` | Testler. |

## Araştırma kuralları

Veri üç döneme ayrılır ve bu ayrım kodla zorlanır:

- **Tasarım** (2024 öncesi): serbestçe denenir, parametreler burada seçilir.
- **Doğrulama** (Ocak 2024 – Haziran 2025): nadiren bakılır, burada ayar yapılmaz.
- **Son sınav** (Temmuz 2025 sonrası): veri yükleyici bu dönemi kendiliğinden
  gizler. Seçilen stratejiler burada yalnızca bir kez denenir. Araştırma
  sırasında `ALGOTRADE_ALLOW_HOLDOUT` değişkeni ayarlanmaz.

Diğer kurallar:

- Her simülasyon `research.run_trial` ile çalıştırılır; böylece atılan denemeler
  de sayılır.
- Her strateji `validate.check_no_lookahead` testinden geçmelidir.
- Günlük mumda üretilen bir karar saatlik ya da dakikalık mumlara verilecekse
  önce `backtest.decisions_on` ile taşınır. Aksi halde karar, bilgi oluşmadan
  bir gün önce dolmuş olur.
- Açığa satış ya da kaldıraç içeren simülasyonlar en yüksek/en düşük fiyatları ve
  fonlama oranlarını ister; mum içi iflas ve fonlama maliyeti böyle hesaplanır.

## Durum

- Veri hattı, simülatör, istatistikler ve deneme kaydı yazıldı ve test edildi.
- Motor dört açıdan bağımsız denetimden geçiyor; bulunan hatalar düzeltildi.
- 1 dakikalık yapı çalışması `research/micro_1m/` altında; raporları oradaki
  `REPORT.md` dosyalarında.
- Sıradaki iş: strateji kolları (spot trend, göreli güç, altın, vadeli
  long/short, fonlama taşıma, saatlik geri dönüş), ardından son sınav ve kâğıt
  işlem botu.

## Canlı işlem için notlar

- Borsada yalnızca işlem sermayesi tutulur; anahtarlar yalnızca al-sat yetkili,
  çekim yetkisiz ve IP'ye bağlı olur.
- Anahtarlar repoya girmez (`.env` git tarafından yok sayılır).
- Canlıya geçmeden önce aynı kod gerçek para olmadan (kâğıt işlem) çalıştırılır.
