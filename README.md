# learnTensorFlow

<h2 style="font-size: 1.8em;text-align: center;color: SaddleBrown;font-family: Verdana, Geneva, Tahoma, sans-serif;"><b> Learn TensorFlow </b></h2>

<h3 style="color: DarkSlateGrey;
    font-size: 1.2em;
    font-family: Verdana, Geneva, Tahoma, sans-serif;"> Setup: </h3>

<div class="setupBoxDiv" style="background-color: Beige;
    border: solid;
    border-width: 1px;
    border-color: cornflowerblue;
    margin: 0px 2px 15px 2px;
    padding: 0px 17px 5px 17px;
    border-radius: 25px;
    font-family: Verdana, Geneva, Tahoma, sans-serif;">
<p>
    <ul>
        <li> 
            <p>vscode (Visual Studio Code) IDE is very useful for this learning. </p>
            <a href="https://code.visualstudio.com/" target="blank_"> Install vscode from https://code.visualstudio.com/ </a>
        </li>
        <li> 
            <p>Add the following extensions in vscode:</p>
            <p>
            <a href="https://marketplace.visualstudio.com/items?itemName=toasty-technologies.octave" target="blank_">Octave</a><br>
            <a href="https://marketplace.visualstudio.com/items?itemName=paulosilva.vsc-octave-debugger" target="blank_">octave-debugger</a><br>
            <a href=https://marketplace.visualstudio.com/items?itemName=ms-python.python" target="blank_">python</a><br>
            <a href=https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode" target="blank_">vscodeintellicode</a><br>
            <a href=https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner" target="blank_">code-runner</a><br>
            </p>
        </li>
        <li>
<p>Also make sure that "python.languageServer" is set to "Jedi" instead of "Microsoft" in vscode settings.
Otherwise memory usage will be around 1GB and it will slow down the PC. vscode team is working on this memory issue. <br>
</p>
<div id="d1" style="color: red;"> 
&emsp; Json settings in vscode: "python.languageServer": "Jedi" </div>
        </li>
    </ul>
</p>
<p>
<ul><li>
Use "pip3 install" to install numpy, pandas, tensorflow and pyplot python modules. 
If you have older python version, make sure that you are using correct version using PATH setting. 
Otherwise, some packages may not work.
</li></ul>
</p>
</div>


<h3 style="color: DarkSlateGrey;
    font-size: 1.2em;
    font-family: Verdana, Geneva, Tahoma, sans-serif;"> Step-1: Get detailed understanding of Machine Learning via Stanford University - Coursera course </h3>

<div class="courseraBoxDiv" style="background-color: LightSkyBlue;
    border: solid;
    border-width: 1px;
    border-color: cornflowerblue;
    margin: 0px 2px 15px 2px;
    padding: 0px 17px 5px 17px;
    border-radius: 25px;
    font-family: Verdana, Geneva, Tahoma, sans-serif;">

<p><ul><li><a href="https://www.coursera.org/learn/machine-learning" target="blank_"><b> <em> Hit Coursera ML Course here</em></b></a><br></li></ul></p>

<p>
<ul><li>
    This course can be completed using 
    <ul>
    <li>Octave (open source tool) </li>
    <li>with some basic mathematical language </li>
    <li>few hours of dedicated time spent/day for few weeks.</li>
    </ul>
</li></ul>
<ul><li>
        Materials and assignments are free in Coursera. 
</ul></li>
<ul><li>
<p>If you pay after completing the course, you will get certificate like this
<a href=https://www.coursera.org/account/accomplishments/verify/YS5P9JM3MJV8?utm_source=link&utm_medium=certificate&utm_content=cert_image&utm_campaign=pdf_header_button&utm_product=course" target="blank_"> <em> certificate reference </em></a>
</ul></li>
</p>
</div>

<h3 style="color: DarkSlateGrey;
    font-size: 1.2em;
    font-family: Verdana, Geneva, Tahoma, sans-serif;"> Step-2: Enroll Machine Learning crash course at Google </h3>

<div class="googleBoxDiv" style="background-color: LightSkyBlue;
    border: solid;
    border-width: 1px;
    border-color: cornflowerblue;
    margin: 0px 2px 15px 2px;
    padding: 0px 17px 5px 17px;
    border-radius: 25px;
    font-family: Verdana, Geneva, Tahoma, sans-serif;">

<p><ul><li> 
<a href="https://developers.google.com/machine-learning/crash-course" target="blank_"><b> <em> Hit Google ML Crash Course here  </em> </b> </a><br>
</li></ul></p>

<ul><li> 
<h5 style="color: DarkSlateGrey;font-size: 0.9em;font-family: Verdana, Geneva, Tahoma, sans-serif;"> Points to note between Coursera and Google ML course: </h5>
</li></ul>
<table frame="box">
<style>
th, td {
    padding: 10px;
    border: 1px solid #666;
}
</style>
<tr>
    <th> Coursera </th>
    <th> Google </th>
    <th> Remarks </th>
</tr>
<tr>
    <td>Theta0</td>
    <td>Bias</td>    
    <td>"Theta0" in Coursera corresponds to "Bias" in Google ML</td>    
</tr>
<tr>
    <td>Theta1...ThetaN</td>
    <td>Weights</td>    
    <td>"Theta1+" in Coursera corresponds to "Weights" in Google ML</td>    
</tr>
<tr>
    <td>Iteration</td>
    <td>Epoch - Iteration</td>    
    <td>Number of iterations per Epoch = Number_of_samples (N) / batch_size (b).<br>
        Each Epoch requires X iterations to cover all N samples with batch_size b.<br>
        For 12 samples with batch_size 3 and Epoch 100, each Epoch will do 4 iterations for 100 times.</td>    
</tr>
<tr>
    <td>J(theta) - Loss vector</td>
    <td>Column "root_mean_squared_error"</td>    
    <td>To get J(theta) in tensorflow, use rmse = hist["root_mean_squared_error"].<br>
        Get hist from history which is returned from "model.fit of tensorFlow".</td>    
</tr>
</table>
</div>

<div class="boxdiv" style="background-color: BurlyWood;
    border: solid;
    border-width: 1px;
    border-color: cornflowerblue;
    margin: 0px 2px 15px 2px;
    padding: 0px 17px 5px 17px;
    border-radius: 25px;
    font-family: Verdana, Geneva, Tahoma, sans-serif;">
<p>
<ul><li><em>
Still in progress...More updates are coming </em> <br><br>
Happy Learning :)
</li></ul>
</p>
</div>
