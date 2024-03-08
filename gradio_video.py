# -*- coding: utf-8 -*-
import gradio as gr
from pre_video2npz import process_video, metadata2numpy
import os
from generate import generate

# 可供选择的选项
number1 = [32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, \
    95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, \
        152, 155, 158, 161, 167, 170, 173, 176, 182, 188, 194, 200, 206, 215, 221, 224]
# param2_options = ["自动提取"]
# for i in number1:
#     param2_options.append(f"手动确定{i}")

# param2_options = [0]  # NOTE
# for i in number1:
#     param2_options.append(i)
    
    
    
# param3_options = [1, 2, 3, 4]   
param3_options = ["快乐兴奋", "愤怒害怕", "悲伤阴郁", "满足希望"]

param4_options = instrument_names = ["Acoustic Grand Piano 大钢琴","Bright Acoustic Piano 明亮的钢琴","Electric Grand Piano 电钢琴","Honky-tonk Piano 酒吧钢琴","Rhodes Piano 柔和的电钢琴",
    "Chorused Piano 加合唱效果的电钢琴","Harpsichord 羽管键琴（拨弦古钢琴）","Clavichord 科拉维科特琴（击弦古钢琴）","Celesta 钢片琴","Glockenspiel","Music box 八音盒","Vibraphone 颤音琴",
    "Marimba 马林巴","Xylophone 木琴","Tubular Bells 管钟","Dulcimer 大扬琴","Hammond Organ 击杆风琴","Percussive Organ 打击式风琴","Rock Organ 摇滚风琴",
    "Church Organ 教堂风琴","Reed Organ 簧管风琴","Accordian 手风琴","Harmonica 口琴","Tango Accordian 探戈手风琴","Acoustic Guitar (nylon) 尼龙弦吉他","Acoustic Guitar (steel) 钢弦吉他",
    "Electric Guitar (jazz) 爵士电吉他","Electric Guitar (clean) 清音电吉他","Electric Guitar (muted) 闷音电吉他","Overdriven Guitar 加驱动效果的电吉他","Distortion Guitar 加失真效果的电吉他",
    "Guitar Harmonics 吉他和音","Acoustic Bass 大贝司（声学贝司）","Electric Bass(finger) 电贝司（指弹）","Electric Bass (pick) 电贝司（拨片）","Fretless Bass 无品贝司",
    "Slap Bass 1 掌击Bass 1","Slap Bass 2 掌击Bass 2","Synth Bass 1 电子合成Bass 1","Synth Bass 2 电子合成Bass 2","Violin 小提琴","Viola 中提琴","Cello 大提琴","Contrabass 低音大提琴",
    "Tremolo Strings 弦乐群颤音音色","Pizzicato Strings 弦乐群拨弦音色","Orchestral Harp 竖琴","Timpani 定音鼓",
    "String Ensemble 1 弦乐合奏音色1","String Ensemble 2 弦乐合奏音色2","Synth Strings 1 合成弦乐合奏音色1","Synth Strings 2 合成弦乐合奏音色2","Choir Aahs 人声合唱“啊”","Voice Oohs 人声“嘟”","Synth Voice 合成人声",
    "Orchestra Hit 管弦乐敲击齐奏","Trumpet 小号","Trombone 长号","Tuba 大号","Muted Trumpet 加弱音器小号","French Horn 法国号（圆号）","Brass Section 铜管组（铜管乐器合奏音色）","Synth Brass 1 合成铜管音色1","Synth Brass 2 合成铜管音色2",
    "Soprano Sax 高音萨克斯风","Alto Sax 次中音萨克斯风","Tenor Sax 中音萨克斯风", "Baritone Sax 低音萨克斯风","Oboe 双簧管",  "English Horn 英国管", "Bassoon 巴松（大管）", "Clarinet 单簧管（黑管）", "Piccolo 短笛",
    "Flute 长笛",  "Recorder 竖笛",  "Pan Flute 排箫",  "Bottle Blow","Shakuhachi 日本尺八",
    "Whistle 口哨声", "Ocarina 奥卡雷那", "Lead 1 (square) 合成主音1（方波）","Lead 2 (sawtooth) 合成主音2（锯齿波）", "Lead 3 (caliope lead) 合成主音3",
    "Lead 4 (chiff lead) 合成主音4","Lead 5 (charang) 合成主音5","Lead 6 (voice) 合成主音6（人声）",
    "Lead 7 (fifths) 合成主音7（平行五度）","Lead 8 (bass+lead)合成主音8（贝司加主音）","Pad 1 (new age) 合成音色1（新世纪）","Pad 2 (warm) 合成音色2 （温暖）","Pad 3 (polysynth) 合成音色3","Pad 4 (choir) 合成音色4 （合唱）",
    "Pad 5 (bowed) 合成音色5", "Pad 6 (metallic) 合成音色6 （金属声）","Pad 7 (halo) 合成音色7 （光环）",
    "Pad 8 (sweep) 合成音色8","FX 1 (rain) 合成效果 1 雨声", "FX 2 (soundtrack) 合成效果 2 音轨","FX 3 (crystal) 合成效果 3 水晶", "FX 4 (atmosphere) 合成效果 4 大气", "FX 5 (brightness) 合成效果 5 明亮",
    "FX 6 (goblins) 合成效果 6 鬼怪","FX 7 (echoes) 合成效果 7 回声", "FX 8 (sci-fi) 合成效果 8 科幻",
    "Sitar 西塔尔（印度）","Banjo 班卓琴（美洲）", "Shamisen 三昧线（日本）", "Koto 十三弦筝（日本）", "Kalimba 卡林巴","Bagpipe 风笛","Fiddle 民族提琴", "Shanai 山奈"
]
dic_param4 = instrument_names = {"Acoustic Grand Piano 大钢琴": 0,"Bright Acoustic Piano 明亮的钢琴": 1,"Electric Grand Piano 电钢琴": 2,"Honky-tonk Piano 酒吧钢琴": 3,"Rhodes Piano 柔和的电钢琴": 4,
    "Chorused Piano 加合唱效果的电钢琴": 5,"Harpsichord 羽管键琴（拨弦古钢琴）": 6,"Clavichord 科拉维科特琴（击弦古钢琴）": 7,"Celesta 钢片琴": 8,"Glockenspiel": 9,"Music box 八音盒": 10,"Vibraphone 颤音琴": 11,
    "Marimba 马林巴": 12,"Xylophone 木琴": 13,"Tubular Bells 管钟": 14,"Dulcimer 大扬琴": 15,"Hammond Organ 击杆风琴": 16,"Percussive Organ 打击式风琴": 17,"Rock Organ 摇滚风琴": 18,
    "Church Organ 教堂风琴": 19,"Reed Organ 簧管风琴": 20,"Accordian 手风琴": 21,"Harmonica 口琴": 22,"Tango Accordian 探戈手风琴": 23,"Acoustic Guitar (nylon) 尼龙弦吉他": 24,"Acoustic Guitar (steel) 钢弦吉他": 25,
    "Electric Guitar (jazz) 爵士电吉他": 26,"Electric Guitar (clean) 清音电吉他": 27,"Electric Guitar (muted) 闷音电吉他": 28,"Overdriven Guitar 加驱动效果的电吉他": 29,"Distortion Guitar 加失真效果的电吉他": 30,
    "Guitar Harmonics 吉他和音": 31,"Acoustic Bass 大贝司（声学贝司）": 32,"Electric Bass(finger) 电贝司（指弹）": 33,"Electric Bass (pick) 电贝司（拨片）": 34,"Fretless Bass 无品贝司": 35,
    "Slap Bass 1 掌击Bass 1": 36,"Slap Bass 2 掌击Bass 2": 37,"Synth Bass 1 电子合成Bass 1": 38,"Synth Bass 2 电子合成Bass 2": 39,"Violin 小提琴": 40,"Viola 中提琴": 41,"Cello 大提琴": 42,"Contrabass 低音大提琴": 43,
    "Tremolo Strings 弦乐群颤音音色": 44,"Pizzicato Strings 弦乐群拨弦音色": 45,"Orchestral Harp 竖琴": 46,"Timpani 定音鼓": 47,
    "String Ensemble 1 弦乐合奏音色1": 48,"String Ensemble 2 弦乐合奏音色2": 49,"Synth Strings 1 合成弦乐合奏音色1": 50,"Synth Strings 2 合成弦乐合奏音色2": 51,"Choir Aahs 人声合唱“啊”": 52,"Voice Oohs 人声“嘟”": 53,"Synth Voice 合成人声": 54,
    "Orchestra Hit 管弦乐敲击齐奏": 55,"Trumpet 小号": 56,"Trombone 长号": 57,"Tuba 大号": 58,"Muted Trumpet 加弱音器小号": 59,"French Horn 法国号（圆号）": 60,"Brass Section 铜管组（铜管乐器合奏音色）": 61,"Synth Brass 1 合成铜管音色1": 62,"Synth Brass 2 合成铜管音色2": 63,
    "Soprano Sax 高音萨克斯风": 64,"Alto Sax 次中音萨克斯风": 65,"Tenor Sax 中音萨克斯风": 66, "Baritone Sax 低音萨克斯风": 67,"Oboe 双簧管": 68,  "English Horn 英国管": 69, "Bassoon 巴松（大管）": 70, "Clarinet 单簧管（黑管）": 71, "Piccolo 短笛": 72,
    "Flute 长笛": 73,  "Recorder 竖笛": 74,  "Pan Flute 排箫": 75,  "Bottle Blow": 76,"Shakuhachi 日本尺八":77,
    "Whistle 口哨声": 78, "Ocarina 奥卡雷那": 79, "Lead 1 (square) 合成主音1（方波）": 80,"Lead 2 (sawtooth) 合成主音2（锯齿波）": 81, "Lead 3 (caliope lead) 合成主音3": 82,
    "Lead 4 (chiff lead) 合成主音4": 83,"Lead 5 (charang) 合成主音5": 84,"Lead 6 (voice) 合成主音6（人声）": 85,
    "Lead 7 (fifths) 合成主音7（平行五度）": 86,"Lead 8 (bass+lead)合成主音8（贝司加主音）": 87,"Pad 1 (new age) 合成音色1（新世纪）": 88,"Pad 2 (warm) 合成音色2 （温暖）": 89,"Pad 3 (polysynth) 合成音色3": 90,"Pad 4 (choir) 合成音色4 （合唱）": 91,
    "Pad 5 (bowed) 合成音色5": 92, "Pad 6 (metallic) 合成音色6 （金属声）": 93,"Pad 7 (halo) 合成音色7 （光环）": 94,
    "Pad 8 (sweep) 合成音色8": 95,"FX 1 (rain) 合成效果 1 雨声": 96, "FX 2 (soundtrack) 合成效果 2 音轨": 97,"FX 3 (crystal) 合成效果 3 水晶": 98, "FX 4 (atmosphere) 合成效果 4 大气": 99, "FX 5 (brightness) 合成效果 5 明亮": 100,
    "FX 6 (goblins) 合成效果 6 鬼怪": 101,"FX 7 (echoes) 合成效果 7 回声": 102, "FX 8 (sci-fi) 合成效果 8 科幻": 103,
    "Sitar 西塔尔（印度）": 104,"Banjo 班卓琴（美洲）": 105, "Shamisen 三昧线（日本）": 106, "Koto 十三弦筝（日本）": 107, "Kalimba 卡林巴": 108,"Bagpipe 风笛": 109,"Fiddle 民族提琴": 110, "Shanai 山奈": 111
}


dic_param3 = {"快乐兴奋": 1, "愤怒害怕": 2, "悲伤阴郁": 3, "满足希望": 4}

param2_options = ["自动提取", "手动确定32", "手动确定35", "手动确定38", "手动确定41", "手动确定44", "手动确定47", "手动确定50", "手动确定53", "手动确定56", "手动确定59", "手动确定62", "手动确定65", "手动确定68", "手动确定71", "手动确定74", "手动确定77", "手动确定80", "手动确定83", "手动确定86", "手动确定89", "手动确定92", \
    "手动确定95", "手动确定98", "手动确定101", "手动确定104", "手动确定107", "手动确定110", "手动确定113", "手动确定116", "手动确定119", "手动确定122", "手动确定125", "手动确定128", "手动确定131", "手动确定134", "手动确定137", "手动确定140", "手动确定143", "手动确定146", "手动确定149", \
        "手动确定152", "手动确定155", "手动确定158", "手动确定161", "手动确定167", "手动确定170", "手动确定173", "手动确定176", "手动确定182", "手动确定188", "手动确定194", "手动确定200", "手动确定206", "手动确定215", "手动确定221", "手动确定224"]

dic_param2 = {"自动提取": 0, "手动确定32": 32, "手动确定35": 35, "手动确定38": 38, "手动确定41": 41, "手动确定44": 44, "手动确定47": 47, "手动确定50": 50, "手动确定53": 53, "手动确定56": 56, "手动确定59": 59, "手动确定62": 62, "手动确定65": 65, "手动确定68": 68, "手动确定71": 71, "手动确定74": 74, "手动确定77": 77, "手动确定80": 80, "手动确定83": 83, "手动确定86": 86, "手动确定89":89, "手动确定92": 92, \
    "手动确定95": 95, "手动确定98": 98, "手动确定101": 101, "手动确定104": 104, "手动确定107": 107, "手动确定110": 110, "手动确定113": 113, "手动确定116": 116, "手动确定119": 119, "手动确定122": 122, "手动确定125": 125, "手动确定128": 128, "手动确定131": 131, "手动确定134": 134, "手动确定137": 137, "手动确定140": 140, "手动确定143": 143, "手动确定146": 146, "手动确定149": 149, \
        "手动确定152": 152, "手动确定155": 155, "手动确定158": 158, "手动确定161": 161, "手动确定167": 167, "手动确定170": 170, "手动确定173": 173, "手动确定176": 176, "手动确定182": 182, "手动确定188": 188, "手动确定194": 194, "手动确定200": 200, "手动确定206": 206, "手动确定215": 215, "手动确定221": 221, "手动确定224": 224}
def process_video_final(input_video, param1, param2, param3, param4):
    out_dir = './inference_our'
    emotion_tag = dic_param3[param3]

    metadata = process_video(input_video, dic_param2[param2])
    video_name = os.path.basename(input_video)
    input_numpy = metadata2numpy(metadata, param1)
    print(input_numpy.dtype)
    generate(input_numpy, input_video, int(emotion_tag), out_dir, dic_param4[param4])
    final_video = os.path.join("./inference_our", os.path.basename(input_video))
    return final_video



# def process_video(input_video, param1, param2, param3, param4):
#     print("param1:", param1)
#     print("param2:", param2)
#     print("param3:", param3)
#     print("param4:", param4)
#     print("type of video is:", type(input_video))

#     return input_video
a = './8426.mp4'

iface = gr.Interface(
    fn=process_video_final, 
    inputs=[gr.inputs.Video(type='mp4', label="上传的视频"),
            gr.inputs.Slider(minimum=0, maximum=1, step=0.001, default=0.2, label="视频节奏提取强度阈值"),
            gr.inputs.Dropdown(param2_options, label="视频节奏提取方式"),
            gr.inputs.Dropdown(param3_options, label="情感类别"),
            gr.inputs.Dropdown(param4_options, label="演奏乐器")],
    outputs=gr.outputs.Video(type='mp4', label="处理后的视频"),
    examples= [[a, 0.2, "手动确定155", "满足希望", "Synth Strings 1 合成弦乐合奏音色1"]]
)

iface.launch(server_name="0.0.0.0",server_port=6002)
