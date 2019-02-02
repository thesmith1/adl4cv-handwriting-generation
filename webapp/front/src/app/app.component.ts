import {Component, OnInit} from '@angular/core';
import {ToBackendService} from './services/to-backend.service';
import {UtilsService} from './services/utils.service';
import {GlobalVars} from './global-vars';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Title of the application';
  private _globalVars: GlobalVars;
  private _text: string;
  private _concurrencyCnt: number;
  public selectedStyle: number;
  public linesBase64: Array<string> = new Array<string>();
  constructor(private _backend: ToBackendService,
              public utils: UtilsService) {
    this._globalVars = new GlobalVars();
  }

  ngOnInit(): void {
    this._text = '';
    this._concurrencyCnt = 0;
    this.selectedStyle = 0;
  }

  public onChange(inputText: string) {
    if (this.linesBase64.length === 0) {
      this.linesBase64.push('');
    }
    this._concurrencyCnt = this._concurrencyCnt + 1;
    this.utils.wait(this._globalVars.keyboardTimeout).then(() => {
      this._concurrencyCnt = this._concurrencyCnt - 1;
      if (this._concurrencyCnt === 0) {
        const index = this._text.length;
        if (index === inputText.length) {
          return;
        }
        this._text = inputText;
        this._backend.generateImageFromString(this._text, Number(this.selectedStyle), index).subscribe((ret) => {
          if (ret['is_new_line']) {
            this.linesBase64[this.linesBase64.length - 1] = ret['completed_line'];
            this.linesBase64.push('');
          }
          this.linesBase64[this.linesBase64.length - 1] = ret['current_line'];
        });
      }
    });
  }
}
