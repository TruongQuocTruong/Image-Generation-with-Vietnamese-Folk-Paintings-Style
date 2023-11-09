from sd_finetune import *
#from hight_resolution import *
######
############
### Streamlit API

st.set_page_config(page_title="TESTING FINE-TUNE STABLE DIFFUSION MODELS", layout="wide",page_icon=":cactus:" )

Init_path = "InputImgs"
Resized_path = "ResizedImgs"
Output_path = "Outputs"

def load_image(image_file):
    img = Image.open(image_file)
    return img

def resize_image(image_path,resized_path):
    # get the basename, e.g. "dragon.jpg" -> ("dragon", ".jpg")
    basename = os.path.splitext(os.path.basename(image_path))[0]
    basetype = os.path.splitext(os.path.basename(image_path))[1]
    with Image.open(image_path) as img:
        # resize the image to 512 x 512
        W = (img.size[0])
        H = (img.size[1])
        #print(W, H)
        if W >= H:
            rate = round(W/H, 2)
            W = 512
            H = int(round(512 / rate, 0))
        else:
            rate = round(H/W, 2)
            H = 512
            W = int(round(512 / rate, 0))   

        size = (W, H)
        img = img.resize(size)
        # rotate the image if required
        # img = img.rotate(90)
        # save the resized image, modify the resample method if required, modify the output directory as well
        print(f"{resized_path}/{basename}{basetype}")
        img.save(f"{resized_path}/{basename}{basetype}", resample=Image.NEAREST)

def clear_cache():
    st.runtime.legacy_caching.clear_cache()


def main():    

    ### Title
    #st.title("TESTING FINE-TUNE STABLE DIFFUSION MODELS")
    st.markdown("<h1 style='text-align: center; color: orange;'>TESTING FINE-TUNE STABLE DIFFUSION MODELS</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: orange;'></h1>", unsafe_allow_html=True)
    #st.markdown("<h1 style='text-align: center; color: white;'></h1>", unsafe_allow_html=True)

    ### Layout
    col1, col2, col3 = st.columns([1,1,0.5], gap="large")

    with col3:
        st.markdown("<h1 style='text-align: center; color: orange; font-size:18px'>SHOWCASE</h1>", unsafe_allow_html=True)

        st.image("ShowCase/DongHo/DH_01.jpg",caption="\"The bird, *Dong Ho painting style, beautiful scenery\"", width=300)
        st.image("ShowCase/SonMai/SM_03.jpg",caption="\"The old quarter in Vietnam with *Son Mai painting style, classic color\"", width =300)

        st.image("ShowCase/DongHo/DH_02.jpg",caption="\"The chicken, *Dong Ho painting style, classic color\"", width =300)
        st.image("ShowCase/SonMai/SM_01.jpg",caption="\"The old quarter in Vietnam with *Son Mai painting style, the sunset\"", width =300)

        st.image("ShowCase/DongHo/DH_03.jpg",caption="\"The KOI fish, *Dong Ho painting style, warm color\"", width =300)
        st.image("ShowCase/DongHo/DH_04.jpg",caption="\"The terraces, *Dong Ho painting style, majestic scenery\"", width =300)
        
        st.image("ShowCase/SonMai/SM_02.jpg",caption="\"The Vietnam's old architecture, *Son Mai painting style, old palace\"", width =300)
        st.image("ShowCase/SonMai/SM_04.jpg",caption="\"The majestic scenery, *Son Mai painting style, the pagoda with old architecture, river, mountain\"", width =300)

    with col1:
        st.markdown("<h1 style='text-align: center; color: orange; font-size:18px'>SETTING</h1>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h1 style='text-align: center; color: orange; font-size:18px'>INPUT IMAGE</h1>", unsafe_allow_html=True)

    with col1:
        with st.form("Form1"):
            ### Input image
            #subtxt1 = st.subheader("Input Init Image:")
            input_img = st.file_uploader("Input Init Image:",type=['png','jpeg','jpg'])
            prompt_caption = st.caption("The sample image to transfer style.")
            if input_img is not None:
                file_details = {
                "FileName":input_img.name,
                "FileType":input_img.type
                }
                #st.write(file_details)
                with open(os.path.join(Init_path,input_img.name),"wb") as f:
                    f.write(input_img.getbuffer())
                #st.success("Saved File")
                with col2:
                    #st.subheader("Input Image")
                    img = load_image(input_img)
                    st.image(img,width=380)

            ### Input promt
            #Take prompt Argument
            prompt = st.text_input("Input Prompt:")
            prompt_caption = st.caption("In the prompt for transfer image style, the style name should be replaced by \" * \" token.")
            prompt_caption = st.caption(" Ex: \"The flower, Dong Ho painting style\"     ===>    \"The flower, *Dong Ho painting style\"")
            
            
            ### Model option
            modelopt = st.selectbox("Model Option:", ("Finetune", "Original"))
            prompt_caption = st.caption("1.An original model was released by Comvis - 2.The fine-tuned model will use a pre-trained original model and fine-tune the embedding part")
            
            
            ### Input strength scale
            embedding = st.selectbox('Choice Version Style:',
                    ("DongHoPaintingStyle", "SonMaiPaintingStyle"))
            prompt_caption = st.caption("The input image will be transfered based on this style. A pre-trained embedding manager checkpoint")
            #Take embedding_path Argument
            embedding_path = "ModelsVer/{fname}/embeddings.pt".format(fname=embedding)
            #embedding_path = "ModelsVer/DongHoPaintingStyle/embeddings.pt"


            ### Size of image option
            #Take size of image
            #scale = st.selectbox('Choice Size Option:', ("Original Size", "Resizied"))
            #W = st.number_input('Insert a width of image', min_value=256, max_value=1024, value =512)
            #H = st.number_input('Insert a height of image', min_value=256, max_value=1024, value = 512)
            #prompt_caption = st.caption("By default: 512 x 512")



            ### Input strength scale
            #Take embedding_path Argument
            strength = st.slider('Input Strength Scale:', 0.0, 1.0, step=0.1, key="slider")
            prompt_caption = st.caption("Better with the 0.4 strength scale. Strength scale for noising / unnoising. 1.0 corresponds to full destruction of information in init image")
            
            submit = st.form_submit_button("Submit")

    with col1:        
        refresh = st.button("Refresh Program", on_click=clear_cache)
        prompt_caption = st.caption("Refresh the cache and reload the model if changing model option!")

    ######
    ############
    ### Call main function
    if submit:
        ### Resized input image to 512x512
        init_img_path = os.path.join(Init_path, input_img.name)
        if os.path.exists(init_img_path):
            #if scale == "Original Size":
                #initimg = init_img_path

            #elif H != 512 or W != 512:
            resize_image(init_img_path, Resized_path)
            #st.write("Resized Image:")
            #temp_resized_img_path = "Resized_{fname}".format(fname=input_img.name)
            #img = load_image(os.path.join(Resized_path,temp_resized_img_path))
            #st.image(img)
            #Take initimg Argument
            initimg = os.path.join(Resized_path,input_img.name)
            #else:
                #resize_image(init_img_path, Resized_path)
                #initimg = os.path.join(Resized_path,input_img.name)

        SD_FINETUNE(initimg, prompt, embedding_path, strength, modelopt)

        with col2:
            #st.subheader("Result Image")
            #Show result
            st.markdown("<h1 style='text-align: center; color: orange; font-size:18px'>RESULT IMAGE</h1>", unsafe_allow_html=True)
            Result_Img = load_image(os.path.join(Output_path,input_img.name))
            st.image(Result_Img, width=380)
            # Download option
            with open(os.path.join(Output_path,input_img.name), "rb") as file:
                #UpSacle = st.selectbox("Up Sacle Image:", ("Hight Superresolution", "Original"))
                #if UpSacle == "Hight Superresolution":
                #HightResolution(os.path.join(Output_path,input_img.name))
                
                st.download_button(label="Download",data=file,
                file_name=os.path.join(Output_path,input_img.name), mime='image/png')



if __name__=='__main__': 
    main()