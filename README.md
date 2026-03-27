**🐱 Tiny Cat Meow Translator**

A Streamlit web app that classifies cat vocalizations into 5 emotion classes using a fine-tuned, pre-trained EfficientNet-B0 model on mel-spectrograms.
[Tiny Cat Meow Translator Link](https://tinycatmeowtranslator.streamlit.app/)

**Data source**: Adapted from the Cat Sound Classification Dataset on Kaggle for academic purposes only: https://www.kaggle.com/datasets/yagtapandeya/cat-sound-classification-dataset/data

**Emotion classes:**
* 😾 Angry:	Cat is annoyed or threatened
* 😸 Happy:	Cat is content and relaxed
* 🙀 Paining:	Cat may be in discomfort
* 😴 Resting:	Cat is calm and at rest
* ⚠️ Warning:	Cat is alerting you to something

**How it works:**
* Upload a `.wav` or `.mp3` (recommended) cat audio clip (up to 10 MB)
* The audio is resampled to 16 kHz, centre-cropped/padded to 5 seconds
* Converted to a 64-band mel-spectrogram (normalized to [0, 255], 3-channel)
* Passed through EfficientNet-B0 fine-tuned on CAT_DB
* The app displays the predicted emotion, confidence breakdown, and mel-spectrogram

**EfficientNet-B0 Fine-tuned Model:**
* Architecture: EfficientNet-B0 pretrained on ImageNet, fine-tuned on an adapted version of [CAT_DB dataset](https://www.kaggle.com/datasets/yagtapandeya/cat-sound-classification-dataset/)
* Input: 64 × 157 × 3 mel-spectrogram (5 s audio @ 16 kHz)
* Output: Softmax over 5 emotion classes
* Framework: TensorFlow / Keras (`.keras` format)

**Preprocessing config**
| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Duration | 5 s |
| n_fft | 1024 |
| hop_length | 512 |
| n_mels | 64 |
| fmin / fmax | 20 / 8000 Hz |
| Normalization | min-max → [0, 255] |

**Limitations**
Training on a tiny dataset of 5 classes * 10 audios = 50 original audios. We augmented the sound to 5 classes * 42 audios/class during the 5-fold CV, and boosted the augmentation to 5 classes * 48 audios/class = 240 audios in the final training model. Test accuracy ~ 90%.

With a huge credit to: 
* https://www.kaggle.com/datasets/yagtapandeya/cat-sound-classification-dataset/
* https://www.kaggle.com/code/muqaddasejaz/cat-emotion-classification-eda
* https://www.kaggle.com/code/whiteant/vggish-cat-sound
* https://github.com/anoojnapaturi42/urban_sound_classifier/
