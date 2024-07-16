from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired


class UploadForm(FlaskForm):
    video = FileField('Upload Video', validators=[DataRequired()])
    submit = SubmitField('Submit')
