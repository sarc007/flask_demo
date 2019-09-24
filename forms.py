from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import StringField,SubmitField, MultipleFileField,FileField
from wtforms.validators import DataRequired,Length, InputRequired
class CustomDemoForm_Train(FlaskForm):
    fname = StringField('First Name',validators=[DataRequired(), Length(min=2, max=25)])
    lname = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=25)])
    trainfileuploads = MultipleFileField('Select Training Files', validators=[Length(min=1), FileAllowed(['jpg', 'png', 'jpeg'])])
    submit = SubmitField('Upload Training Files')
class CustomDemoForm_Test(FlaskForm):
    testfileupload = FileField('Select Test Image File To Identify The Face Inside It', validators=[FileAllowed(['jpg', 'png', 'jpeg', 'mp4', 'avi'])])
    submit = SubmitField('Upload Test File')
